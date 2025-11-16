// Function: sub_215CE20
// Address: 0x215ce20
//
void *__fastcall sub_215CE20(__int64 a1)
{
  bool v1; // zf
  __int64 v2; // rax
  __int64 v3; // rax
  _QWORD *v5; // rbx
  _QWORD *v6; // r14
  __int64 v7; // rax
  _QWORD *v8; // rbx
  _QWORD *v9; // r14
  __int64 v10; // rax
  __int64 v11; // rax
  _QWORD *v12; // rbx
  _QWORD *v13; // r12
  __int64 v14; // rsi
  __int64 v15; // rax
  _QWORD *v16; // rbx
  _QWORD *v17; // r12
  __int64 v18; // rsi
  __int64 v19; // [rsp+8h] [rbp-88h] BYREF
  __int64 v20; // [rsp+10h] [rbp-80h]
  __int64 v21; // [rsp+18h] [rbp-78h]
  __int64 v22; // [rsp+20h] [rbp-70h]
  void *v23; // [rsp+30h] [rbp-60h]
  __int64 v24; // [rsp+38h] [rbp-58h] BYREF
  __int64 v25; // [rsp+40h] [rbp-50h]
  __int64 v26; // [rsp+48h] [rbp-48h]
  __int64 v27; // [rsp+50h] [rbp-40h]

  v1 = *(_BYTE *)(a1 + 304) == 0;
  *(_QWORD *)a1 = off_4A01A88;
  if ( !v1 )
  {
    v11 = *(unsigned int *)(a1 + 296);
    if ( (_DWORD)v11 )
    {
      v12 = *(_QWORD **)(a1 + 280);
      v13 = &v12[2 * v11];
      do
      {
        if ( *v12 != -8 && *v12 != -4 )
        {
          v14 = v12[1];
          if ( v14 )
            sub_161E7C0((__int64)(v12 + 1), v14);
        }
        v12 += 2;
      }
      while ( v13 != v12 );
    }
    j___libc_free_0(*(_QWORD *)(a1 + 280));
  }
  v2 = *(unsigned int *)(a1 + 264);
  if ( (_DWORD)v2 )
  {
    v5 = *(_QWORD **)(a1 + 248);
    v19 = 2;
    v20 = 0;
    v21 = -8;
    v6 = &v5[6 * v2];
    v22 = 0;
    v24 = 2;
    v25 = 0;
    v26 = -16;
    v23 = &unk_4A01B30;
    v27 = 0;
    do
    {
      v7 = v5[3];
      *v5 = &unk_49EE2B0;
      if ( v7 != 0 && v7 != -8 && v7 != -16 )
        sub_1649B30(v5 + 1);
      v5 += 6;
    }
    while ( v6 != v5 );
    v23 = &unk_49EE2B0;
    if ( v26 != -8 && v26 != 0 && v26 != -16 )
      sub_1649B30(&v24);
    if ( v21 != -8 && v21 != 0 && v21 != -16 )
      sub_1649B30(&v19);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 248));
  if ( *(_BYTE *)(a1 + 224) )
  {
    v15 = *(unsigned int *)(a1 + 216);
    if ( (_DWORD)v15 )
    {
      v16 = *(_QWORD **)(a1 + 200);
      v17 = &v16[2 * v15];
      do
      {
        if ( *v16 != -8 && *v16 != -4 )
        {
          v18 = v16[1];
          if ( v18 )
            sub_161E7C0((__int64)(v16 + 1), v18);
        }
        v16 += 2;
      }
      while ( v17 != v16 );
    }
    j___libc_free_0(*(_QWORD *)(a1 + 200));
  }
  v3 = *(unsigned int *)(a1 + 184);
  if ( (_DWORD)v3 )
  {
    v8 = *(_QWORD **)(a1 + 168);
    v19 = 2;
    v20 = 0;
    v21 = -8;
    v9 = &v8[6 * v3];
    v22 = 0;
    v24 = 2;
    v25 = 0;
    v26 = -16;
    v23 = &unk_49F8530;
    v27 = 0;
    do
    {
      v10 = v8[3];
      *v8 = &unk_49EE2B0;
      if ( v10 != 0 && v10 != -8 && v10 != -16 )
        sub_1649B30(v8 + 1);
      v8 += 6;
    }
    while ( v9 != v8 );
    v23 = &unk_49EE2B0;
    if ( v26 != -8 && v26 != 0 && v26 != -16 )
      sub_1649B30(&v24);
    if ( v21 != -8 && v21 != 0 && v21 != -16 )
      sub_1649B30(&v19);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 168));
  return sub_1636790((_QWORD *)a1);
}
