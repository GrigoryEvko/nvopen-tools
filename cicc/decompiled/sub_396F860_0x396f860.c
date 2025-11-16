// Function: sub_396F860
// Address: 0x396f860
//
__int64 __fastcall sub_396F860(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r14
  unsigned __int8 v6; // al
  _QWORD *v7; // rdi
  __int64 v8; // rsi
  __int64 result; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rcx
  _QWORD *v21; // rcx
  __int64 v22; // rax
  __int64 ***v23; // r14
  __int64 **v24; // rax
  __int64 v25; // rax
  __int64 v26; // rbx
  __int64 v27; // r15
  unsigned __int64 v28; // r12
  unsigned __int64 v29; // rbx
  char v30; // al
  unsigned __int64 v31; // rax
  int v32; // edi
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  int v36; // ebx
  __int64 v37; // rdi
  unsigned __int64 v38; // rax
  __int64 v39; // rax
  unsigned int v40; // edx
  __int64 v41; // r8
  unsigned __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // r12
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 *v47; // [rsp+8h] [rbp-B8h]
  __int64 v48; // [rsp+8h] [rbp-B8h]
  __int64 v49; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v50; // [rsp+18h] [rbp-A8h] BYREF
  unsigned __int64 v51; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v52; // [rsp+28h] [rbp-98h]
  unsigned __int64 v53; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v54; // [rsp+38h] [rbp-88h]
  unsigned __int64 v55; // [rsp+40h] [rbp-80h] BYREF
  __int64 v56; // [rsp+48h] [rbp-78h]
  char v57; // [rsp+50h] [rbp-70h] BYREF
  unsigned __int64 v58; // [rsp+60h] [rbp-60h] BYREF
  __int64 v59; // [rsp+68h] [rbp-58h]
  __int64 v60; // [rsp+70h] [rbp-50h]
  __int64 v61; // [rsp+78h] [rbp-48h]
  int v62; // [rsp+80h] [rbp-40h]
  unsigned __int64 *v63; // [rsp+88h] [rbp-38h]

  v5 = a1[31];
  if ( sub_1593BB0(a2, a2, a3, a4) || (v6 = *(_BYTE *)(a2 + 16), v6 == 9) )
  {
    v8 = v5;
    v7 = 0;
    return sub_38CB470((__int64)v7, v8);
  }
  if ( v6 == 13 )
  {
    v7 = *(_QWORD **)(a2 + 24);
    if ( *(_DWORD *)(a2 + 32) > 0x40u )
      v7 = (_QWORD *)*v7;
    v8 = v5;
    return sub_38CB470((__int64)v7, v8);
  }
  if ( v6 <= 3u )
  {
    v10 = sub_396EAF0((__int64)a1, a2);
    return sub_38CF310(v10, 0, v5, 0);
  }
  if ( v6 == 4 )
  {
    v10 = sub_396F840((__int64)a1, a2);
    return sub_38CF310(v10, 0, v5, 0);
  }
  switch ( *(_WORD *)(a2 + 18) )
  {
    case 0xB:
    case 0xF:
    case 0x12:
    case 0x15:
    case 0x17:
    case 0x1A:
    case 0x1B:
    case 0x1C:
      goto LABEL_26;
    case 0xD:
      v52 = 1;
      v51 = 0;
      v14 = sub_396DDB0((__int64)a1);
      if ( !(unsigned __int8)sub_14D5D40(
                               *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)),
                               &v49,
                               (__int64)&v51,
                               v14) )
        goto LABEL_23;
      v54 = 1;
      v53 = 0;
      v15 = sub_396DDB0((__int64)a1);
      if ( (unsigned __int8)sub_14D5D40(
                              *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))),
                              &v50,
                              (__int64)&v53,
                              v15) )
      {
        v39 = sub_396DD80((__int64)a1);
        v28 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)v39 + 120LL))(
                v39,
                v49,
                v50,
                a1[29]);
        if ( !v28 )
        {
          v43 = sub_396EAF0((__int64)a1, v50);
          v44 = sub_38CF310(v43, 0, v5, 0);
          v45 = sub_396EAF0((__int64)a1, v49);
          v46 = sub_38CF310(v45, 0, v5, 0);
          v28 = sub_38CB1F0(17, v46, v44, v5, 0);
        }
        LODWORD(v56) = v52;
        if ( v52 > 0x40 )
          sub_16A4FD0((__int64)&v55, (const void **)&v51);
        else
          v55 = v51;
        sub_16A7590((__int64)&v55, (__int64 *)&v53);
        v40 = v56;
        LODWORD(v56) = 0;
        LODWORD(v59) = v40;
        v58 = v55;
        if ( v40 > 0x40 )
          v41 = *(_QWORD *)v55;
        else
          v41 = (__int64)(v55 << (64 - (unsigned __int8)v40)) >> (64 - (unsigned __int8)v40);
        v48 = v41;
        sub_135E100((__int64 *)&v58);
        sub_135E100((__int64 *)&v55);
        if ( v48 )
        {
          v42 = sub_38CB470(v48, v5);
          v28 = sub_38CB1F0(0, v28, v42, v5, 0);
        }
        sub_135E100((__int64 *)&v53);
        sub_135E100((__int64 *)&v51);
        goto LABEL_44;
      }
      if ( v54 > 0x40 && v53 )
        j_j___libc_free_0_0(v53);
LABEL_23:
      if ( v52 > 0x40 && v51 )
        j_j___libc_free_0_0(v51);
LABEL_26:
      v16 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD))(*a1 + 216LL))(
              a1,
              *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      v17 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD))(*a1 + 216LL))(
              a1,
              *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))));
      v18 = v16;
      v19 = v17;
      v20 = v5;
      switch ( *(_WORD *)(a2 + 18) )
      {
        case 0xB:
          v32 = 0;
          break;
        case 0xC:
        case 0xE:
        case 0x10:
        case 0x11:
        case 0x13:
        case 0x14:
        case 0x16:
        case 0x18:
        case 0x19:
        case 0x1C:
          v32 = 18;
          break;
        case 0xD:
          v32 = 17;
          break;
        case 0xF:
          v32 = 11;
          break;
        case 0x12:
          v32 = 2;
          break;
        case 0x15:
          v32 = 10;
          break;
        case 0x17:
          v32 = 14;
          break;
        case 0x1A:
          v32 = 1;
          break;
        case 0x1B:
          v32 = 13;
          break;
      }
      return sub_38CB1F0(v32, v18, v19, v20, 0);
    case 0x20:
      v33 = sub_396DDB0((__int64)a1);
      LODWORD(v59) = sub_15A9570(v33, *(_QWORD *)a2);
      if ( (unsigned int)v59 > 0x40 )
        sub_16A4EF0((__int64)&v58, 0, 0);
      else
        v58 = 0;
      v34 = sub_396DDB0((__int64)a1);
      sub_1634900(a2, v34, (__int64)&v58);
      v35 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD))(*a1 + 216LL))(
              a1,
              *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      v36 = v59;
      v28 = v35;
      if ( (unsigned int)v59 <= 0x40 )
      {
        if ( !v58 )
          goto LABEL_44;
        v37 = (__int64)(v58 << (64 - (unsigned __int8)v59)) >> (64 - (unsigned __int8)v59);
      }
      else
      {
        if ( v36 == (unsigned int)sub_16A57B0((__int64)&v58) )
          goto LABEL_54;
        v37 = *(_QWORD *)v58;
      }
      v38 = sub_38CB470(v37, v5);
      v28 = sub_38CB1F0(0, v28, v38, v5, 0);
      if ( (unsigned int)v59 <= 0x40 )
        goto LABEL_44;
LABEL_54:
      if ( v58 )
        j_j___libc_free_0_0(v58);
LABEL_44:
      result = v28;
      break;
    case 0x24:
    case 0x2F:
      goto LABEL_27;
    case 0x2D:
      v25 = sub_396DDB0((__int64)a1);
      v26 = *(_QWORD *)a2;
      v27 = v25;
      v47 = *(__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      v28 = (*(__int64 (__fastcall **)(_QWORD *, __int64 *))(*a1 + 216LL))(a1, v47);
      v29 = sub_12BE0A0(v27, v26);
      if ( v29 == sub_12BE0A0(v27, *v47) )
        goto LABEL_44;
      v30 = sub_12BE0A0(v27, *v47);
      v31 = sub_38CB470(0xFFFFFFFFFFFFFFFFLL >> (64 - 8 * v30), v5);
      v20 = v5;
      v18 = v28;
      v19 = v31;
      v32 = 1;
      return sub_38CB1F0(v32, v18, v19, v20, 0);
    case 0x2E:
      v22 = sub_396DDB0((__int64)a1);
      v23 = *(__int64 ****)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      v24 = (__int64 **)sub_15A9650(v22, *(_QWORD *)a2);
      v13 = sub_15A4750(v23, v24, 0);
      return (*(__int64 (__fastcall **)(_QWORD *, __int64))(*a1 + 216LL))(a1, v13);
    case 0x30:
      if ( *(_DWORD *)(*(_QWORD *)a2 + 8LL) >> 8 )
        goto LABEL_29;
LABEL_27:
      v13 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      return (*(__int64 (__fastcall **)(_QWORD *, __int64))(*a1 + 216LL))(a1, v13);
    default:
      v11 = sub_396DDB0((__int64)a1);
      v12 = sub_14DBA30(a2, v11, 0);
      v13 = v12;
      if ( v12 == a2 || !v12 )
      {
LABEL_29:
        v56 = 0;
        v55 = (unsigned __int64)&v57;
        v57 = 0;
        v62 = 1;
        v61 = 0;
        v60 = 0;
        v59 = 0;
        v58 = (unsigned __int64)&unk_49EFBE0;
        v63 = &v55;
        sub_1263B40((__int64)&v58, "Unsupported expression in static initializer: ");
        v21 = (_QWORD *)a1[33];
        if ( v21 )
          v21 = *(_QWORD **)(*v21 + 40LL);
        sub_15537D0(a2, (__int64)&v58, 0, v21);
        if ( v61 != v59 )
          sub_16E7BA0((__int64 *)&v58);
        sub_16BD160((__int64)v63, 1u);
      }
      return (*(__int64 (__fastcall **)(_QWORD *, __int64))(*a1 + 216LL))(a1, v13);
  }
  return result;
}
