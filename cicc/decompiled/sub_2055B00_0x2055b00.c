// Function: sub_2055B00
// Address: 0x2055b00
//
__int64 __fastcall sub_2055B00(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // r10
  __int64 v9; // r14
  bool v10; // cf
  unsigned __int64 v11; // rax
  __int64 v12; // r11
  __int64 v13; // r15
  __int64 v14; // r11
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // r9
  __int64 v19; // rsi
  __int64 v20; // rsi
  __int64 v21; // rax
  int v22; // eax
  __int64 i; // r14
  __int64 v24; // rsi
  __int64 v26; // r15
  __int64 v27; // rax
  __int64 v28; // [rsp+8h] [rbp-58h]
  __int64 v29; // [rsp+8h] [rbp-58h]
  __int64 v30; // [rsp+10h] [rbp-50h]
  __int64 v31; // [rsp+18h] [rbp-48h]
  __int64 v32; // [rsp+20h] [rbp-40h]
  __int64 v33; // [rsp+20h] [rbp-40h]
  __int64 v34; // [rsp+28h] [rbp-38h]
  __int64 v35; // [rsp+28h] [rbp-38h]

  v4 = a1[1];
  v5 = *a1;
  v6 = 0xCCCCCCCCCCCCCCCDLL * ((v4 - *a1) >> 4);
  if ( v6 == 0x199999999999999LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0xCCCCCCCCCCCCCCCDLL * ((a1[1] - *a1) >> 4);
  v8 = a2;
  v9 = a2;
  v10 = __CFADD__(v7, v6);
  v11 = v7 - 0x3333333333333333LL * ((a1[1] - *a1) >> 4);
  v12 = a2 - v5;
  if ( v10 )
  {
    v26 = 0x7FFFFFFFFFFFFFD0LL;
  }
  else
  {
    if ( !v11 )
    {
      v30 = 0;
      v13 = 80;
      v31 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0x199999999999999LL )
      v11 = 0x199999999999999LL;
    v26 = 80 * v11;
  }
  v29 = a3;
  v27 = sub_22077B0(v26);
  v12 = a2 - v5;
  v8 = a2;
  v31 = v27;
  a3 = v29;
  v30 = v27 + v26;
  v13 = v27 + 80;
LABEL_7:
  v14 = v31 + v12;
  if ( v14 )
  {
    v15 = *(_QWORD *)(a3 + 56);
    *(_DWORD *)v14 = *(_DWORD *)a3;
    v16 = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(v14 + 56) = v15;
    *(_QWORD *)(v14 + 8) = v16;
    *(_QWORD *)(v14 + 16) = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(v14 + 24) = *(_QWORD *)(a3 + 24);
    *(_QWORD *)(v14 + 32) = *(_QWORD *)(a3 + 32);
    *(_QWORD *)(v14 + 40) = *(_QWORD *)(a3 + 40);
    *(_QWORD *)(v14 + 48) = *(_QWORD *)(a3 + 48);
    if ( v15 )
    {
      v28 = a3;
      v32 = v8;
      v34 = v14;
      sub_1623A60(v14 + 56, v15, 2);
      a3 = v28;
      v8 = v32;
      v14 = v34;
    }
    *(_DWORD *)(v14 + 64) = *(_DWORD *)(a3 + 64);
    *(_DWORD *)(v14 + 72) = *(_DWORD *)(a3 + 72);
    *(_DWORD *)(v14 + 76) = *(_DWORD *)(a3 + 76);
  }
  if ( v8 != v5 )
  {
    v17 = v31;
    v18 = v5;
    while ( 1 )
    {
      if ( v17 )
      {
        *(_DWORD *)v17 = *(_DWORD *)v18;
        *(_QWORD *)(v17 + 8) = *(_QWORD *)(v18 + 8);
        *(_QWORD *)(v17 + 16) = *(_QWORD *)(v18 + 16);
        *(_QWORD *)(v17 + 24) = *(_QWORD *)(v18 + 24);
        *(_QWORD *)(v17 + 32) = *(_QWORD *)(v18 + 32);
        *(_QWORD *)(v17 + 40) = *(_QWORD *)(v18 + 40);
        *(_QWORD *)(v17 + 48) = *(_QWORD *)(v18 + 48);
        v19 = *(_QWORD *)(v18 + 56);
        *(_QWORD *)(v17 + 56) = v19;
        if ( v19 )
        {
          v33 = v8;
          v35 = v18;
          sub_1623A60(v17 + 56, v19, 2);
          v8 = v33;
          v18 = v35;
        }
        *(_DWORD *)(v17 + 64) = *(_DWORD *)(v18 + 64);
        *(_DWORD *)(v17 + 72) = *(_DWORD *)(v18 + 72);
        *(_DWORD *)(v17 + 76) = *(_DWORD *)(v18 + 76);
      }
      v18 += 80;
      if ( v8 == v18 )
        break;
      v17 += 80;
    }
    v13 = v17 + 160;
  }
  if ( v8 != v4 )
  {
    do
    {
      v20 = *(_QWORD *)(v9 + 56);
      *(_DWORD *)v13 = *(_DWORD *)v9;
      v21 = *(_QWORD *)(v9 + 8);
      *(_QWORD *)(v13 + 56) = v20;
      *(_QWORD *)(v13 + 8) = v21;
      *(_QWORD *)(v13 + 16) = *(_QWORD *)(v9 + 16);
      *(_QWORD *)(v13 + 24) = *(_QWORD *)(v9 + 24);
      *(_QWORD *)(v13 + 32) = *(_QWORD *)(v9 + 32);
      *(_QWORD *)(v13 + 40) = *(_QWORD *)(v9 + 40);
      *(_QWORD *)(v13 + 48) = *(_QWORD *)(v9 + 48);
      if ( v20 )
        sub_1623A60(v13 + 56, v20, 2);
      v22 = *(_DWORD *)(v9 + 64);
      v9 += 80;
      v13 += 80;
      *(_DWORD *)(v13 - 16) = v22;
      *(_DWORD *)(v13 - 8) = *(_DWORD *)(v9 - 8);
      *(_DWORD *)(v13 - 4) = *(_DWORD *)(v9 - 4);
    }
    while ( v4 != v9 );
  }
  for ( i = v5; i != v4; i += 80 )
  {
    v24 = *(_QWORD *)(i + 56);
    if ( v24 )
      sub_161E7C0(i + 56, v24);
  }
  if ( v5 )
    j_j___libc_free_0(v5, a1[2] - v5);
  a1[1] = v13;
  *a1 = v31;
  a1[2] = v30;
  return v30;
}
