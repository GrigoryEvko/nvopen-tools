// Function: sub_2495360
// Address: 0x2495360
//
__int64 __fastcall sub_2495360(
        _QWORD *a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  unsigned __int64 v8; // r13
  char v9; // al
  __int64 v10; // rax
  unsigned __int64 *v11; // r12
  __int64 **v12; // rax
  __int64 v13; // rax
  __int64 v14; // r11
  __int64 **v15; // rax
  unsigned int **v16; // r11
  __int64 v17; // rax
  unsigned int v18; // r13d
  __int64 v19; // rax
  __int64 v21; // [rsp+8h] [rbp-C8h]
  __int64 v22; // [rsp+8h] [rbp-C8h]
  unsigned int **v23; // [rsp+8h] [rbp-C8h]
  unsigned int v26; // [rsp+28h] [rbp-A8h]
  _QWORD v27[4]; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v28; // [rsp+50h] [rbp-80h]
  _QWORD v29[4]; // [rsp+60h] [rbp-70h] BYREF
  __int64 v30; // [rsp+80h] [rbp-50h]
  __int64 v31; // [rsp+88h] [rbp-48h]
  __int64 v32; // [rsp+90h] [rbp-40h]

  v8 = a2;
  v9 = *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL);
  switch ( v9 )
  {
    case 2:
      v11 = (unsigned __int64 *)(*a1 + 200LL);
      break;
    case 3:
      v11 = (unsigned __int64 *)(*a1 + 216LL);
      break;
    case 4:
      v10 = *a1;
      v21 = a1[3];
      v11 = (unsigned __int64 *)(*a1 + 216LL);
      LOWORD(v30) = 257;
      v12 = (__int64 **)sub_BCB170(*(_QWORD **)(v10 + 8));
      v27[0] = v26;
      if ( *(_BYTE *)(v21 + 108) )
        v8 = sub_B358C0(v21, 0x71u, a2, (__int64)v12, v26, (__int64)v29, 0, 0, 0);
      else
        v8 = sub_24932B0((__int64 *)v21, 0x2Du, a2, v12, (__int64)v29, 0, v26, 0);
      v13 = *a1;
      v14 = a1[3];
      LOWORD(v30) = 257;
      v22 = v14;
      v15 = (__int64 **)sub_BCB170(*(_QWORD **)(v13 + 8));
      v27[0] = v26;
      if ( *(_BYTE *)(v22 + 108) )
        a3 = sub_B358C0(v22, 0x71u, v8, (__int64)v15, v26, (__int64)v29, 0, 0, 0);
      else
        a3 = sub_24932B0((__int64 *)v22, 0x2Du, v8, v15, (__int64)v29, 0, v26, 0);
      break;
    default:
      BUG();
  }
  v29[1] = a3;
  v16 = (unsigned int **)a1[3];
  v28 = 257;
  v29[2] = a4;
  v29[3] = a5;
  v17 = a1[1];
  v29[0] = v8;
  v23 = v16;
  v18 = *(_WORD *)(v17 + 2) & 0x3F;
  v19 = sub_BCB2D0(*(_QWORD **)(a1[2] + 72LL));
  v30 = sub_ACD640(v19, v18, 0);
  v31 = a6;
  v32 = a7;
  return sub_921880(v23, *v11, v11[1], (int)v29, 7, (__int64)v27, 0);
}
