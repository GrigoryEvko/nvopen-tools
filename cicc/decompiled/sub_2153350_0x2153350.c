// Function: sub_2153350
// Address: 0x2153350
//
__int64 __fastcall sub_2153350(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  unsigned __int8 v5; // bl
  __int64 v6; // r14
  unsigned __int8 v7; // al
  _QWORD *v8; // rdi
  __int64 v9; // rsi
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 ***v18; // r14
  __int64 **v19; // rax
  unsigned int v20; // eax
  __int64 v21; // r8
  __int64 v22; // rax
  int v23; // ebx
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // r15
  __int64 v27; // rax
  __int64 v28; // r15
  unsigned __int64 v29; // rbx
  char v30; // al
  __int64 v31; // rax
  _QWORD *v32; // rcx
  __int64 v33; // [rsp+0h] [rbp-90h]
  __int64 v34; // [rsp+8h] [rbp-88h]
  __int64 *v35; // [rsp+8h] [rbp-88h]
  _QWORD v36[2]; // [rsp+10h] [rbp-80h] BYREF
  char v37; // [rsp+20h] [rbp-70h] BYREF
  __int64 *v38; // [rsp+30h] [rbp-60h] BYREF
  __int64 v39; // [rsp+38h] [rbp-58h]
  __int64 v40; // [rsp+40h] [rbp-50h]
  __int64 v41; // [rsp+48h] [rbp-48h]
  int v42; // [rsp+50h] [rbp-40h]
  _QWORD *v43; // [rsp+58h] [rbp-38h]

  while ( 2 )
  {
    v4 = a2;
    v5 = a3;
    v6 = *(_QWORD *)(a1 + 248);
    if ( sub_1593BB0(a2, a2, a3, a4) || (v7 = *(_BYTE *)(a2 + 16), v7 == 9) )
    {
      v9 = v6;
      v8 = 0;
      return sub_38CB470(v8, v9);
    }
    if ( v7 == 13 )
    {
      v8 = *(_QWORD **)(a2 + 24);
      if ( *(_DWORD *)(a2 + 32) > 0x40u )
        v8 = (_QWORD *)*v8;
      v9 = v6;
      return sub_38CB470(v8, v9);
    }
    if ( v7 > 3u )
    {
      switch ( *(_WORD *)(a2 + 18) )
      {
        case 0xB:
          v26 = sub_2153350(a1, *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), v5);
          v27 = sub_2153350(a1, *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))), v5);
          return sub_38CB1F0(0, v26, v27, v6, 0);
        case 0x20:
          v34 = sub_396DDB0(a1);
          v20 = sub_15A9570(v34, *(_QWORD *)a2);
          v21 = v34;
          LODWORD(v39) = v20;
          if ( v20 > 0x40 )
          {
            sub_16A4EF0((__int64)&v38, 0, 0);
            v21 = v34;
          }
          else
          {
            v38 = 0;
          }
          sub_1634900(a2, v21, (__int64)&v38);
          v22 = sub_2153350(a1, *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), v5);
          v23 = v39;
          v13 = v22;
          if ( (unsigned int)v39 <= 0x40 )
          {
            if ( !v38 )
              return v13;
            v24 = (__int64)((_QWORD)v38 << (64 - (unsigned __int8)v39)) >> (64 - (unsigned __int8)v39);
          }
          else
          {
            if ( v23 == (unsigned int)sub_16A57B0((__int64)&v38) )
              goto LABEL_29;
            v24 = *v38;
          }
          v25 = sub_38CB470(v24, v6);
          v13 = sub_38CB1F0(0, v13, v25, v6, 0);
          if ( (unsigned int)v39 <= 0x40 )
            return v13;
LABEL_29:
          if ( v38 )
            j_j___libc_free_0_0(v38);
          break;
        case 0x24:
        case 0x2F:
          a3 = v5;
          v16 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
          a4 = 4 * v16;
          a2 = *(_QWORD *)(a2 - 24 * v16);
          continue;
        case 0x2D:
          v28 = sub_396DDB0(a1);
          v33 = *(_QWORD *)a2;
          v35 = *(__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
          v13 = sub_2153350(a1, v35, v5);
          v29 = sub_12BE0A0(v28, v33);
          if ( v29 == sub_12BE0A0(v28, *v35) )
            return v13;
          v30 = sub_12BE0A0(v28, *v35);
          v31 = sub_38CB470(0xFFFFFFFFFFFFFFFFLL >> (64 - 8 * v30), v6);
          return sub_38CB1F0(1, v13, v31, v6, 0);
        case 0x2E:
          v17 = sub_396DDB0(a1);
          v18 = *(__int64 ****)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
          v19 = (__int64 **)sub_15A9650(v17, *(_QWORD *)a2);
          a2 = sub_15A4750(v18, v19, 0);
          goto LABEL_17;
        case 0x30:
          if ( *(_DWORD *)(*(_QWORD *)a2 + 8LL) >> 8 )
            goto LABEL_37;
          a3 = 1;
          a2 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
          continue;
        default:
          v14 = sub_396DDB0(a1);
          v15 = sub_14DBA30(a2, v14, 0);
          a2 = v15;
          if ( !v15 || v15 == v4 )
          {
LABEL_37:
            v36[1] = 0;
            v36[0] = &v37;
            v37 = 0;
            v42 = 1;
            v41 = 0;
            v40 = 0;
            v39 = 0;
            v38 = (__int64 *)&unk_49EFBE0;
            v43 = v36;
            sub_1263B40((__int64)&v38, "Unsupported expression in static initializer: ");
            v32 = *(_QWORD **)(a1 + 264);
            if ( v32 )
              v32 = *(_QWORD **)(*v32 + 40LL);
            sub_15537D0(v4, (__int64)&v38, 0, v32);
            if ( v41 != v39 )
              sub_16E7BA0((__int64 *)&v38);
            sub_16BD160((__int64)v43, 1u);
          }
LABEL_17:
          a3 = v5;
          continue;
      }
    }
    else
    {
      v11 = sub_396EAF0(a1, a2);
      v12 = sub_38CF310(v11, 0, v6, 0);
      v13 = v12;
      if ( v5 )
      {
        v13 = sub_2163120(v12, v6);
        if ( v13 )
          v13 += 8;
      }
    }
    return v13;
  }
}
