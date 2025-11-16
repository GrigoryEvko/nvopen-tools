// Function: sub_970A00
// Address: 0x970a00
//
__int64 __fastcall sub_970A00(__int64 a1, unsigned __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rax
  __int64 v4; // r14
  _BYTE *v5; // rbx
  char v6; // r15
  __int64 v7; // rax
  size_t v8; // rdx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdx
  _BYTE *v11; // r9
  _BYTE *v12; // rax
  __int64 v13; // rdx
  char v14; // r8
  __int64 v15; // rax
  size_t v16; // r12
  void *v17; // r13
  __int64 v18; // rax
  __int64 v19; // [rsp-170h] [rbp-170h]
  __int64 v20; // [rsp-170h] [rbp-170h]
  unsigned __int64 v21; // [rsp-170h] [rbp-170h]
  unsigned __int64 v22; // [rsp-168h] [rbp-168h] BYREF
  char v23; // [rsp-160h] [rbp-160h]
  _BYTE *v24; // [rsp-158h] [rbp-158h] BYREF
  size_t v25; // [rsp-150h] [rbp-150h]
  __int64 v26; // [rsp-148h] [rbp-148h]
  _BYTE v27[320]; // [rsp-140h] [rbp-140h] BYREF

  if ( (*(_BYTE *)(a1 + 80) & 1) == 0 )
    return 0;
  if ( (unsigned __int8)sub_B2FC80(a1) )
    return 0;
  if ( (unsigned __int8)sub_B2F6B0(a1) )
    return 0;
  if ( (*(_BYTE *)(a1 + 80) & 2) != 0 )
    return 0;
  v3 = sub_B2F730(a1);
  v4 = *(_QWORD *)(a1 - 32);
  v5 = (_BYTE *)v3;
  v19 = *(_QWORD *)(v4 + 8);
  v6 = sub_AE5020(v3, v19);
  v7 = sub_9208B0((__int64)v5, v19);
  v25 = v8;
  v24 = (_BYTE *)v7;
  v22 = ((1LL << v6) + ((unsigned __int64)(v7 + 7) >> 3) - 1) >> v6 << v6;
  v23 = v8;
  if ( sub_CA1930(&v22) < a2 )
    return 0;
  v9 = sub_CA1930(&v22) - a2;
  v10 = v9;
  if ( v9 > 0xFFFF )
    return 0;
  v25 = 0;
  v24 = v27;
  v11 = v27;
  v26 = 256;
  if ( v9 )
  {
    v12 = v27;
    if ( v10 > 0x100 )
    {
      v21 = v10;
      sub_C8D290(&v24, v27, v10, 1);
      v11 = v24;
      v10 = v21;
      v12 = &v24[v25];
    }
    if ( v12 != &v11[v10] )
    {
      do
      {
        if ( v12 )
          *v12 = 0;
        ++v12;
      }
      while ( &v11[v10] != v12 );
      v11 = v24;
    }
    v25 = v10;
  }
  v14 = sub_9704C0((unsigned __int8 *)v4, a2, (__int64)v11, v10, v5);
  result = 0;
  if ( v14 )
  {
    v15 = sub_BD5C60(a1, a2, v13);
    v16 = v25;
    v17 = v24;
    v18 = sub_BCD140(v15, 8);
    sub_BCD420(v18, v16);
    a2 = v16;
    result = sub_AC9630(v17, v16);
  }
  if ( v24 != v27 )
  {
    v20 = result;
    _libc_free(v24, a2);
    return v20;
  }
  return result;
}
