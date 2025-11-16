// Function: sub_12A8C20
// Address: 0x12a8c20
//
__int64 __fastcall sub_12A8C20(_QWORD *a1, __int64 a2, _BYTE *a3)
{
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 i; // rbx
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // rdi
  __int64 *v13; // r15
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rsi
  __int64 v17; // rsi
  __int64 v18; // rax
  _QWORD *v19; // r15
  __int64 v20; // rdi
  unsigned __int64 *v21; // r13
  __int64 v22; // rax
  unsigned __int64 v23; // rcx
  __int64 v24; // rsi
  __int64 v25; // rsi
  __int64 v27; // [rsp+20h] [rbp-D0h]
  __int64 v29; // [rsp+30h] [rbp-C0h]
  __int64 v30; // [rsp+38h] [rbp-B8h]
  __int64 v31; // [rsp+40h] [rbp-B0h]
  __int64 v32; // [rsp+58h] [rbp-98h] BYREF
  char v33[16]; // [rsp+60h] [rbp-90h] BYREF
  __int16 v34; // [rsp+70h] [rbp-80h]
  char v35[16]; // [rsp+80h] [rbp-70h] BYREF
  __int16 v36; // [rsp+90h] [rbp-60h]
  _QWORD v37[2]; // [rsp+A0h] [rbp-50h] BYREF
  __int16 v38; // [rsp+B0h] [rbp-40h]

  v4 = sub_1599EF0(a2);
  v29 = sub_1643350(a1[5]);
  v30 = **(_QWORD **)(a2 + 16);
  v5 = *(_QWORD *)(a2 + 32);
  if ( (_DWORD)v5 )
  {
    v27 = (unsigned int)v5;
    for ( i = 0; i != v27; ++i )
    {
      while ( 1 )
      {
        v38 = 257;
        v7 = sub_12A8800(a1 + 6, v30, a3, i, (__int64)v37);
        v36 = 257;
        v8 = v7;
        v9 = sub_15A0680(v29, i, 0);
        v34 = 257;
        v31 = v9;
        v10 = sub_1648A60(64, 1);
        v11 = v10;
        if ( v10 )
          sub_15F9210(v10, v30, v8, 0, 0, 0);
        v12 = a1[7];
        if ( v12 )
        {
          v13 = (__int64 *)a1[8];
          sub_157E9D0(v12 + 40, v11);
          v14 = *(_QWORD *)(v11 + 24);
          v15 = *v13;
          *(_QWORD *)(v11 + 32) = v13;
          v15 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v11 + 24) = v15 | v14 & 7;
          *(_QWORD *)(v15 + 8) = v11 + 24;
          *v13 = *v13 & 7 | (v11 + 24);
        }
        sub_164B780(v11, v33);
        v16 = a1[6];
        if ( v16 )
        {
          v37[0] = a1[6];
          sub_1623A60(v37, v16, 2);
          if ( *(_QWORD *)(v11 + 48) )
            sub_161E7C0(v11 + 48);
          v17 = v37[0];
          *(_QWORD *)(v11 + 48) = v37[0];
          if ( v17 )
            sub_1623210(v37, v17, v11 + 48);
        }
        if ( *(_BYTE *)(v4 + 16) > 0x10u || *(_BYTE *)(v11 + 16) > 0x10u || *(_BYTE *)(v31 + 16) > 0x10u )
          break;
        ++i;
        v4 = sub_15A3890(v4, v11, v31, 0);
        if ( v27 == i )
          return v4;
      }
      v38 = 257;
      v18 = sub_1648A60(56, 3);
      v19 = (_QWORD *)v18;
      if ( v18 )
        sub_15FA480(v18, v4, v11, v31, v37, 0);
      v20 = a1[7];
      if ( v20 )
      {
        v21 = (unsigned __int64 *)a1[8];
        sub_157E9D0(v20 + 40, v19);
        v22 = v19[3];
        v23 = *v21;
        v19[4] = v21;
        v23 &= 0xFFFFFFFFFFFFFFF8LL;
        v19[3] = v23 | v22 & 7;
        *(_QWORD *)(v23 + 8) = v19 + 3;
        *v21 = *v21 & 7 | (unsigned __int64)(v19 + 3);
      }
      sub_164B780(v19, v35);
      v24 = a1[6];
      if ( v24 )
      {
        v32 = a1[6];
        sub_1623A60(&v32, v24, 2);
        if ( v19[6] )
          sub_161E7C0(v19 + 6);
        v25 = v32;
        v19[6] = v32;
        if ( v25 )
          sub_1623210(&v32, v25, v19 + 6);
      }
      v4 = (__int64)v19;
    }
  }
  return v4;
}
