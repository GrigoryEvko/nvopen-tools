// Function: sub_20CC690
// Address: 0x20cc690
//
__int64 __fastcall sub_20CC690(int a1, __int64 *a2, __int64 a3, __int64 a4, double a5, double a6, double a7)
{
  unsigned __int16 v10; // si
  __int64 v11; // rax
  _QWORD *v12; // r12
  bool v14; // cc
  __int64 v15; // rdi
  __int64 v16; // rax
  unsigned __int64 *v17; // r13
  __int64 v18; // rdi
  unsigned __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 *v22; // r13
  __int64 v23; // r14
  __int64 v24; // rsi
  unsigned __int8 *v25; // rsi
  __int64 v26; // rdx
  __int64 *v27; // rsi
  int v28; // edi
  __int64 v29; // rax
  __int64 v30; // rdi
  __int64 v31; // rax
  unsigned __int64 *v32; // r13
  __int64 v33; // rax
  unsigned __int64 v34; // rcx
  __int64 v35; // rsi
  __int64 v36; // rsi
  __int64 v37; // [rsp+8h] [rbp-88h] BYREF
  _QWORD v38[2]; // [rsp+10h] [rbp-80h] BYREF
  __int16 v39; // [rsp+20h] [rbp-70h]
  __int64 v40[2]; // [rsp+30h] [rbp-60h] BYREF
  char v41; // [rsp+40h] [rbp-50h]
  char v42; // [rsp+41h] [rbp-4Fh]
  _QWORD v43[2]; // [rsp+50h] [rbp-40h] BYREF
  __int16 v44; // [rsp+60h] [rbp-30h]

  switch ( a1 )
  {
    case 0:
      return a4;
    case 1:
      v14 = *(_BYTE *)(a3 + 16) <= 0x10u;
      v42 = 1;
      v40[0] = (__int64)"new";
      v41 = 3;
      if ( !v14 || *(_BYTE *)(a4 + 16) > 0x10u )
      {
        v26 = a4;
        v44 = 257;
        v27 = (__int64 *)a3;
        v28 = 11;
        goto LABEL_34;
      }
      return sub_15A2B30((__int64 *)a3, a4, 0, 0, a5, a6, a7);
    case 2:
      v14 = *(_BYTE *)(a3 + 16) <= 0x10u;
      v42 = 1;
      v40[0] = (__int64)"new";
      v41 = 3;
      if ( !v14 || *(_BYTE *)(a4 + 16) > 0x10u )
      {
        v26 = a4;
        v44 = 257;
        v27 = (__int64 *)a3;
        v28 = 13;
        goto LABEL_34;
      }
      return sub_15A2B60((__int64 *)a3, a4, 0, 0, a5, a6, a7);
    case 3:
      v43[0] = "new";
      v44 = 259;
      return sub_1281C00(a2, a3, a4, (__int64)v43);
    case 4:
      v42 = 1;
      v40[0] = (__int64)"new";
      v41 = 3;
      v39 = 257;
      v15 = sub_1281C00(a2, a3, a4, (__int64)v38);
      if ( *(_BYTE *)(v15 + 16) <= 0x10u )
        return sub_15A2B00((__int64 *)v15, a5, a6, a7);
      v44 = 257;
      v12 = (_QWORD *)sub_15FB630((__int64 *)v15, (__int64)v43, 0);
      v31 = a2[1];
      if ( v31 )
      {
        v32 = (unsigned __int64 *)a2[2];
        sub_157E9D0(v31 + 40, (__int64)v12);
        v33 = v12[3];
        v34 = *v32;
        v12[4] = v32;
        v34 &= 0xFFFFFFFFFFFFFFF8LL;
        v12[3] = v34 | v33 & 7;
        *(_QWORD *)(v34 + 8) = v12 + 3;
        *v32 = *v32 & 7 | (unsigned __int64)(v12 + 3);
      }
      sub_164B780((__int64)v12, v40);
      v35 = *a2;
      if ( *a2 )
      {
        v22 = &v37;
        v23 = (__int64)(v12 + 6);
        v37 = *a2;
        sub_1623A60((__int64)&v37, v35, 2);
        v36 = v12[6];
        if ( v36 )
          sub_161E7C0((__int64)(v12 + 6), v36);
        v25 = (unsigned __int8 *)v37;
        v12[6] = v37;
        if ( v25 )
          goto LABEL_30;
      }
      return (__int64)v12;
    case 5:
      v14 = *(_BYTE *)(a4 + 16) <= 0x10u;
      v42 = 1;
      v40[0] = (__int64)"new";
      v41 = 3;
      if ( !v14 )
        goto LABEL_37;
      v12 = (_QWORD *)a3;
      if ( !sub_1593BB0(a4, (__int64)a2, a3, a4) )
      {
        if ( *(_BYTE *)(a3 + 16) > 0x10u )
        {
LABEL_37:
          v26 = a4;
          v27 = (__int64 *)a3;
          v44 = 257;
          v28 = 27;
LABEL_34:
          v29 = sub_15FB440(v28, v27, v26, (__int64)v43, 0);
          v30 = a2[1];
          v12 = (_QWORD *)v29;
          if ( v30 )
          {
            v17 = (unsigned __int64 *)a2[2];
            v18 = v30 + 40;
            goto LABEL_25;
          }
          goto LABEL_26;
        }
        v12 = (_QWORD *)sub_15A2D10((__int64 *)a3, a4, a5, a6, a7);
      }
      break;
    case 6:
      v14 = *(_BYTE *)(a3 + 16) <= 0x10u;
      v42 = 1;
      v40[0] = (__int64)"new";
      v41 = 3;
      if ( !v14
        || *(_BYTE *)(a4 + 16) > 0x10u
        || (v12 = (_QWORD *)sub_15A2A30((__int64 *)0x1C, (__int64 *)a3, a4, 0, 0, a5, a6, a7)) == 0 )
      {
        v44 = 257;
        v12 = (_QWORD *)sub_15FB440(28, (__int64 *)a3, a4, (__int64)v43, 0);
        v16 = a2[1];
        if ( v16 )
        {
          v17 = (unsigned __int64 *)a2[2];
          v18 = v16 + 40;
LABEL_25:
          sub_157E9D0(v18, (__int64)v12);
          v19 = *v17;
          v20 = v12[3];
          v12[4] = v17;
          v19 &= 0xFFFFFFFFFFFFFFF8LL;
          v12[3] = v19 | v20 & 7;
          *(_QWORD *)(v19 + 8) = v12 + 3;
          *v17 = *v17 & 7 | (unsigned __int64)(v12 + 3);
        }
LABEL_26:
        sub_164B780((__int64)v12, v40);
        v21 = *a2;
        if ( *a2 )
        {
          v22 = v38;
          v38[0] = *a2;
          v23 = (__int64)(v12 + 6);
          sub_1623A60((__int64)v38, v21, 2);
          v24 = v12[6];
          if ( v24 )
            sub_161E7C0((__int64)(v12 + 6), v24);
          v25 = (unsigned __int8 *)v38[0];
          v12[6] = v38[0];
          if ( v25 )
LABEL_30:
            sub_1623210((__int64)v22, v25, v23);
        }
      }
      return (__int64)v12;
    case 7:
      v44 = 257;
      v10 = 38;
      goto LABEL_3;
    case 8:
      v10 = 41;
      v44 = 257;
      goto LABEL_3;
    case 9:
      v10 = 34;
      v44 = 257;
      goto LABEL_3;
    case 10:
      v10 = 37;
      v44 = 257;
LABEL_3:
      v11 = sub_12AA0C0(a2, v10, (_BYTE *)a3, a4, (__int64)v43);
      v43[0] = "new";
      v44 = 259;
      v12 = (_QWORD *)sub_156B790(a2, v11, a3, a4, (__int64)v43, 0);
      break;
  }
  return (__int64)v12;
}
