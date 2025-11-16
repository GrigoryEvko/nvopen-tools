// Function: sub_31E0ED0
// Address: 0x31e0ed0
//
__int64 __fastcall sub_31E0ED0(_QWORD *a1, unsigned __int8 *a2)
{
  unsigned __int8 *v3; // r12
  _QWORD *v4; // r14
  int v5; // edx
  _QWORD *v6; // rsi
  _QWORD *v7; // rdi
  unsigned __int64 (__fastcall *v9)(__int64, __int64); // rax
  __int64 v10; // rax
  unsigned __int64 v11; // r15
  __int64 v12; // rdi
  __int64 (*v13)(); // rax
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // r14
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // r14
  __int64 v22; // rax
  unsigned int v23; // edx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 *v27; // rbx
  __int64 v28; // rax
  __int64 v29; // rax
  int v30; // r12d
  __int64 v31; // rax
  unsigned __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // r15
  __int64 v35; // rax
  __int64 v36; // r15
  __int64 v37; // rax
  __int64 v38; // r14
  __int64 v39; // r15
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rcx
  __int64 v43; // rax
  int v44; // eax
  __int64 v45; // r12
  __int64 v46; // rax
  unsigned __int64 v47; // r12
  __int64 v48; // rax
  unsigned __int64 v49; // rax
  unsigned __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // [rsp+8h] [rbp-F8h]
  __int64 v53; // [rsp+18h] [rbp-E8h] BYREF
  __int64 v54; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v55; // [rsp+28h] [rbp-D8h] BYREF
  __int64 v56; // [rsp+30h] [rbp-D0h] BYREF
  unsigned int v57; // [rsp+38h] [rbp-C8h]
  char *v58; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v59; // [rsp+48h] [rbp-B8h]
  char v60; // [rsp+50h] [rbp-B0h] BYREF
  __int64 *v61; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v62; // [rsp+68h] [rbp-98h]
  __int16 v63; // [rsp+80h] [rbp-80h]
  __int64 *v64; // [rsp+90h] [rbp-70h] BYREF
  __int64 v65; // [rsp+98h] [rbp-68h]
  __int64 v66; // [rsp+A0h] [rbp-60h]
  __int64 v67; // [rsp+A8h] [rbp-58h]
  char *v68; // [rsp+B0h] [rbp-50h]
  __int64 v69; // [rsp+B8h] [rbp-48h]
  char **v70; // [rsp+C0h] [rbp-40h]

  v3 = a2;
  v4 = (_QWORD *)a1[27];
  if ( sub_AC30F0((__int64)a2) || (v5 = *a2, (unsigned int)(v5 - 12) <= 1) )
  {
    v6 = v4;
    v7 = 0;
    return sub_E81A90((__int64)v7, v6, 0, 0);
  }
  if ( (_BYTE)v5 == 17 )
  {
    v7 = (_QWORD *)*((_QWORD *)a2 + 3);
    if ( *((_DWORD *)a2 + 8) > 0x40u )
      v7 = (_QWORD *)*v7;
    v6 = v4;
    return sub_E81A90((__int64)v7, v6, 0, 0);
  }
  if ( (_BYTE)v5 != 8 )
  {
    if ( (unsigned __int8)v5 <= 3u )
    {
LABEL_14:
      v10 = sub_31DB510((__int64)a1, (__int64)a2);
      return sub_E808D0(v10, 0, v4, 0);
    }
    switch ( (_BYTE)v5 )
    {
      case 4:
        v9 = *(unsigned __int64 (__fastcall **)(__int64, __int64))(*a1 + 368LL);
        if ( v9 != sub_31E0EA0 )
          return v9((__int64)a1, (__int64)a2);
        v4 = (_QWORD *)a1[27];
        v10 = sub_31E0E80((__int64)a1, (__int64)a2);
        return sub_E808D0(v10, 0, v4, 0);
      case 6:
        v11 = 0;
        v12 = sub_31DA6B0((__int64)a1);
        v13 = *(__int64 (**)())(*(_QWORD *)v12 + 184LL);
        if ( v13 != sub_302E450 )
          return ((__int64 (__fastcall *)(__int64, unsigned __int8 *, _QWORD))v13)(v12, a2, a1[25]);
        return v11;
      case 7:
        a2 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
        goto LABEL_14;
    }
    if ( (_BYTE)v5 != 5 )
      BUG();
    switch ( *((_WORD *)a2 + 1) )
    {
      case 0xD:
        v36 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD))(*a1 + 240LL))(
                a1,
                *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)]);
        v37 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD))(*a1 + 240LL))(
                a1,
                *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))]);
        return sub_E81A00(0, v36, v37, v4, 0);
      case 0xF:
        v27 = &v56;
        v57 = 1;
        v56 = 0;
        v33 = sub_31DA930((__int64)a1);
        if ( !(unsigned __int8)sub_96E080(
                                 *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)],
                                 &v53,
                                 (__int64)&v56,
                                 v33,
                                 &v54) )
          goto LABEL_43;
        LODWORD(v59) = 1;
        v58 = 0;
        v41 = sub_31DA930((__int64)a1);
        if ( !(unsigned __int8)sub_96E080(
                                 *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))],
                                 &v55,
                                 (__int64)&v58,
                                 v41,
                                 0) )
        {
          sub_969240((__int64 *)&v58);
LABEL_43:
          v34 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD))(*a1 + 240LL))(
                  a1,
                  *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)]);
          v35 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD))(*a1 + 240LL))(
                  a1,
                  *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))]);
          v11 = sub_E81A00(18, v34, v35, v4, 0);
          goto LABEL_41;
        }
        v43 = sub_31DA6B0((__int64)a1);
        v11 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)v43 + 176LL))(
                v43,
                v53,
                v55,
                a1[25]);
        if ( !v11 )
        {
          v46 = sub_31DB510((__int64)a1, v53);
          v47 = sub_E808D0(v46, 0, v4, 0);
          if ( v54 && *(_BYTE *)(sub_31DA6B0((__int64)a1) + 939) )
          {
            v51 = sub_31DA6B0((__int64)a1);
            v47 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v51 + 184LL))(v51, v54, a1[25]);
          }
          v48 = sub_31DB510((__int64)a1, v55);
          v49 = sub_E808D0(v48, 0, v4, 0);
          v11 = sub_E81A00(18, v47, v49, v4, 0);
        }
        LODWORD(v62) = v57;
        if ( v57 > 0x40 )
          sub_C43780((__int64)&v61, (const void **)&v56);
        else
          v61 = (__int64 *)v56;
        sub_C46B40((__int64)&v61, (__int64 *)&v58);
        v44 = v62;
        LODWORD(v62) = 0;
        LODWORD(v65) = v44;
        v64 = v61;
        v45 = sub_31D9190((__int64 *)&v64);
        sub_969240((__int64 *)&v64);
        sub_969240((__int64 *)&v61);
        if ( v45 )
        {
          v50 = sub_E81A90(v45, v4, 0, 0);
          v11 = sub_E81A00(0, v11, v50, v4, 0);
        }
        sub_969240((__int64 *)&v58);
        goto LABEL_41;
      case 0x22:
        v26 = sub_31DA930((__int64)a1);
        LODWORD(v65) = sub_AE43A0(v26, *((_QWORD *)a2 + 1));
        if ( (unsigned int)v65 > 0x40 )
        {
          v27 = (__int64 *)&v64;
          sub_C43690((__int64)&v64, 0, 0);
        }
        else
        {
          v64 = 0;
          v27 = (__int64 *)&v64;
        }
        v28 = sub_31DA930((__int64)a1);
        sub_BB6360((__int64)a2, v28, (__int64)&v64, 0, 0);
        v29 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD))(*a1 + 240LL))(
                a1,
                *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)]);
        v30 = v65;
        v11 = v29;
        if ( (unsigned int)v65 <= 0x40 )
        {
          if ( !v64 )
          {
LABEL_41:
            sub_969240(v27);
            return v11;
          }
        }
        else if ( v30 == (unsigned int)sub_C444A0((__int64)&v64) )
        {
          goto LABEL_41;
        }
        v31 = sub_31D9190((__int64 *)&v64);
        v32 = sub_E81A90(v31, v4, 0, 0);
        v11 = sub_E81A00(0, v11, v32, v4, 0);
        goto LABEL_41;
      case 0x26:
      case 0x31:
        return (*(__int64 (__fastcall **)(_QWORD *, _QWORD))(*a1 + 240LL))(
                 a1,
                 *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)]);
      case 0x2F:
        v14 = sub_31DA930((__int64)a1);
        v15 = *((_QWORD *)a2 + 1);
        v16 = v14;
        v52 = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        v11 = (*(__int64 (__fastcall **)(_QWORD *, __int64))(*a1 + 240LL))(a1, v52);
        v61 = (__int64 *)sub_BDB740(v16, v15);
        v62 = v17;
        v64 = (__int64 *)sub_BDB740(v16, *(_QWORD *)(v52 + 8));
        v65 = v18;
        if ( v64 >= v61 )
          return v11;
        goto LABEL_27;
      case 0x30:
        v38 = sub_31DA930((__int64)a1);
        v39 = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        v40 = sub_AE4450(v38, *((_QWORD *)a2 + 1));
        v20 = sub_96F3F0(v39, v40, 0, v38);
        if ( !v20 )
          goto LABEL_27;
        return (*(__int64 (__fastcall **)(_QWORD *, __int64))(*a1 + 240LL))(a1, v20);
      case 0x32:
        v21 = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        v22 = *((_QWORD *)a2 + 1);
        if ( (unsigned int)*(unsigned __int8 *)(v22 + 8) - 17 <= 1 )
          v22 = **(_QWORD **)(v22 + 16);
        v23 = *(_DWORD *)(v22 + 8);
        v24 = *(_QWORD *)(v21 + 8);
        v25 = v23 >> 8;
        if ( (unsigned int)*(unsigned __int8 *)(v24 + 8) - 17 <= 1 )
          v24 = **(_QWORD **)(v24 + 16);
        if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD, __int64))(*(_QWORD *)a1[25] + 80LL))(
                a1[25],
                *(_DWORD *)(v24 + 8) >> 8,
                v25) )
          goto LABEL_27;
        return (*(__int64 (__fastcall **)(_QWORD *, __int64))(*a1 + 240LL))(a1, v21);
      default:
LABEL_27:
        v19 = sub_31DA930((__int64)a1);
        v20 = sub_97B670(v3, v19, 0);
        if ( (unsigned __int8 *)v20 == v3 )
        {
          v58 = &v60;
          v70 = &v58;
          v69 = 0x100000000LL;
          v59 = 0;
          v64 = (__int64 *)&unk_49DD210;
          v60 = 0;
          v65 = 0;
          v66 = 0;
          v67 = 0;
          v68 = 0;
          sub_CB5980((__int64)&v64, 0, 0, 0);
          if ( (unsigned __int64)(v67 - (_QWORD)v68) <= 0x2D )
          {
            sub_CB6200((__int64)&v64, "Unsupported expression in static initializer: ", 0x2Eu);
          }
          else
          {
            qmemcpy(v68, "Unsupported expression in static initializer: ", 0x2Eu);
            v68 += 46;
          }
          v42 = a1[29];
          if ( v42 )
            v42 = *(_QWORD *)(*(_QWORD *)v42 + 40LL);
          sub_A5BF40(v3, (__int64)&v64, 0, v42);
          v63 = 260;
          v61 = (__int64 *)&v58;
          sub_C64D30((__int64)&v61, 1u);
        }
        return (*(__int64 (__fastcall **)(_QWORD *, __int64))(*a1 + 240LL))(a1, v20);
    }
  }
  v9 = *(unsigned __int64 (__fastcall **)(__int64, __int64))(*a1 + 360LL);
  return v9((__int64)a1, (__int64)a2);
}
