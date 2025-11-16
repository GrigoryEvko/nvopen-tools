// Function: sub_BA72D0
// Address: 0xba72d0
//
__int64 __fastcall sub_BA72D0(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  unsigned __int8 v4; // al
  __int64 *v5; // r15
  __int64 v6; // rdx
  __int64 *v7; // r14
  __int64 v8; // rdx
  unsigned __int8 v9; // al
  __int64 *v10; // r15
  __int64 v11; // rdx
  __int64 *v12; // r14
  __int64 v13; // rdx
  __int64 *v14; // rsi
  __int64 *v15; // rdi
  __int64 v17; // [rsp+18h] [rbp-88h] BYREF
  __int64 v18; // [rsp+20h] [rbp-80h] BYREF
  __int64 v19; // [rsp+28h] [rbp-78h]
  __int64 v20; // [rsp+30h] [rbp-70h]
  __int64 v21; // [rsp+38h] [rbp-68h]
  __int64 *v22; // [rsp+40h] [rbp-60h]
  __int64 v23; // [rsp+48h] [rbp-58h]
  _BYTE v24[80]; // [rsp+50h] [rbp-50h] BYREF

  if ( !a1 )
    return a2;
  v3 = a1;
  if ( a2 )
  {
    v4 = *(_BYTE *)(a1 - 16);
    if ( (v4 & 2) != 0 )
    {
      v5 = *(__int64 **)(a1 - 32);
      v6 = *(unsigned int *)(a1 - 24);
    }
    else
    {
      v5 = (__int64 *)(a1 - 8LL * ((v4 >> 2) & 0xF) - 16);
      v6 = (*(_WORD *)(a1 - 16) >> 6) & 0xF;
    }
    v7 = &v5[v6];
    v18 = 0;
    v22 = (__int64 *)v24;
    v19 = 0;
    v20 = 0;
    v21 = 0;
    v23 = 0x400000000LL;
    while ( v5 != v7 )
    {
      v8 = *v5++;
      v17 = v8;
      sub_BA67F0((__int64)&v18, &v17);
    }
    v9 = *(_BYTE *)(a2 - 16);
    if ( (v9 & 2) != 0 )
    {
      v10 = *(__int64 **)(a2 - 32);
      v11 = *(unsigned int *)(a2 - 24);
    }
    else
    {
      v10 = (__int64 *)(a2 - 8LL * ((v9 >> 2) & 0xF) - 16);
      v11 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
    }
    v12 = &v10[v11];
    while ( v10 != v12 )
    {
      v13 = *v10++;
      v17 = v13;
      sub_BA67F0((__int64)&v18, &v17);
    }
    v14 = v22;
    v15 = (__int64 *)(*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL);
    if ( (*(_QWORD *)(a1 + 8) & 4) != 0 )
      v15 = (__int64 *)*v15;
    v3 = sub_B9D9A0(v15, v22, (__int64 *)(unsigned int)v23);
    if ( v22 != (__int64 *)v24 )
      _libc_free(v22, v14);
    sub_C7D6A0(v19, 8LL * (unsigned int)v21, 8);
  }
  return v3;
}
