// Function: sub_1286000
// Address: 0x1286000
//
__int64 __fastcall sub_1286000(_QWORD *a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // rax
  _QWORD *v5; // r13
  _QWORD *v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdi
  unsigned __int64 *v16; // r14
  __int64 v17; // rax
  unsigned __int64 v18; // rcx
  __int64 v19; // rsi
  _QWORD *v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rdx
  __int64 v28; // rsi
  __int64 *v29; // [rsp+10h] [rbp-90h]
  __int64 *v30; // [rsp+18h] [rbp-88h]
  __int64 v31; // [rsp+28h] [rbp-78h] BYREF
  _QWORD v32[2]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v33; // [rsp+40h] [rbp-60h]
  _BYTE v34[16]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v35; // [rsp+60h] [rbp-40h]

  v3 = a1 + 6;
  v5 = a2;
  v7 = (_QWORD *)a1[4];
  v8 = *a2;
  v33 = 257;
  v30 = v3;
  if ( v7[91] != v8 )
  {
    if ( *((_BYTE *)a2 + 16) > 0x10u )
    {
      v13 = v7[91];
      v35 = 257;
      v14 = sub_15FDBD0(47, a2, v13, v34, 0);
      v15 = a1[7];
      v5 = (_QWORD *)v14;
      if ( v15 )
      {
        v16 = (unsigned __int64 *)a1[8];
        sub_157E9D0(v15 + 40, v14);
        v17 = v5[3];
        v18 = *v16;
        v5[4] = v16;
        v18 &= 0xFFFFFFFFFFFFFFF8LL;
        v5[3] = v18 | v17 & 7;
        *(_QWORD *)(v18 + 8) = v5 + 3;
        *v16 = *v16 & 7 | (unsigned __int64)(v5 + 3);
      }
      sub_164B780(v5, v32);
      v19 = a1[6];
      if ( v19 )
      {
        v31 = a1[6];
        sub_1623A60(&v31, v19, 2);
        v20 = v5 + 6;
        if ( v5[6] )
        {
          sub_161E7C0(v5 + 6);
          v20 = v5 + 6;
        }
        v21 = v31;
        v5[6] = v31;
        if ( v21 )
          sub_1623210(&v31, v21, v20);
      }
      v7 = (_QWORD *)a1[4];
      v8 = v7[91];
    }
    else
    {
      v9 = sub_15A46C0(47, a2, v7[91], 0);
      v7 = (_QWORD *)a1[4];
      v5 = (_QWORD *)v9;
      v8 = v7[91];
    }
  }
  v33 = 257;
  if ( *(_QWORD *)a3 != v8 )
  {
    if ( *(_BYTE *)(a3 + 16) > 0x10u )
    {
      v35 = 257;
      v22 = sub_15FDBD0(47, a3, v8, v34, 0);
      v23 = a1[7];
      a3 = v22;
      if ( v23 )
      {
        v29 = (__int64 *)a1[8];
        sub_157E9D0(v23 + 40, v22);
        v24 = *v29;
        v25 = *(_QWORD *)(a3 + 24) & 7LL;
        *(_QWORD *)(a3 + 32) = v29;
        v24 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(a3 + 24) = v24 | v25;
        *(_QWORD *)(v24 + 8) = a3 + 24;
        *v29 = *v29 & 7 | (a3 + 24);
      }
      sub_164B780(a3, v32);
      v26 = a1[6];
      if ( v26 )
      {
        v31 = a1[6];
        sub_1623A60(&v31, v26, 2);
        v27 = a3 + 48;
        if ( *(_QWORD *)(a3 + 48) )
        {
          sub_161E7C0(a3 + 48);
          v27 = a3 + 48;
        }
        v28 = v31;
        *(_QWORD *)(a3 + 48) = v31;
        if ( v28 )
          sub_1623210(&v31, v28, v27);
      }
      v7 = (_QWORD *)a1[4];
    }
    else
    {
      v10 = sub_15A46C0(47, a3, v8, 0);
      v7 = (_QWORD *)a1[4];
      a3 = v10;
    }
  }
  v32[0] = v5;
  v32[1] = a3;
  v35 = 257;
  v11 = sub_126A190(v7, 212, 0, 0);
  return sub_1285290(v30, *(_QWORD *)(v11 + 24), v11, (int)v32, 2, (__int64)v34, 0);
}
