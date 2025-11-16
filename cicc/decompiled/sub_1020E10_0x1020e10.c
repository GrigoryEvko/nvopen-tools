// Function: sub_1020E10
// Address: 0x1020e10
//
__int64 __fastcall sub_1020E10(__int64 a1, const __m128i *a2, _QWORD a3, _QWORD a4, _QWORD a5, __int64 a6)
{
  __int64 v7; // rax
  unsigned __int8 **v8; // rbx
  unsigned __int8 **v9; // r12
  __int64 v10; // r8
  unsigned __int8 **v11; // rax
  int v12; // edx
  unsigned __int8 **v13; // rsi
  unsigned __int8 *v14; // rax
  __int64 v15; // r12
  __int64 v17; // [rsp+8h] [rbp-88h]
  unsigned __int8 **v18; // [rsp+10h] [rbp-80h] BYREF
  __int64 v19; // [rsp+18h] [rbp-78h]
  _BYTE v20[112]; // [rsp+20h] [rbp-70h] BYREF

  v7 = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    v8 = *(unsigned __int8 ***)(a1 - 8);
    v9 = &v8[(unsigned __int64)v7 / 8];
  }
  else
  {
    v9 = (unsigned __int8 **)a1;
    v8 = (unsigned __int8 **)(a1 - v7);
  }
  v18 = (unsigned __int8 **)v20;
  v10 = v7 >> 5;
  v19 = 0x800000000LL;
  if ( (unsigned __int64)v7 > 0x100 )
  {
    v17 = v7 >> 5;
    sub_C8D5F0((__int64)&v18, v20, v7 >> 5, 8u, v10, a6);
    v13 = v18;
    v12 = v19;
    LODWORD(v10) = v17;
    v11 = &v18[(unsigned int)v19];
  }
  else
  {
    v11 = (unsigned __int8 **)v20;
    v12 = 0;
    v13 = (unsigned __int8 **)v20;
  }
  if ( v8 != v9 )
  {
    do
    {
      if ( v11 )
        *v11 = *v8;
      v8 += 4;
      ++v11;
    }
    while ( v8 != v9 );
    v13 = v18;
    v12 = v19;
  }
  LODWORD(v19) = v10 + v12;
  v14 = sub_100EA40((unsigned __int8 *)a1, v13, (unsigned int)(v10 + v12), a2, 3, a6);
  v15 = (__int64)v14;
  if ( (unsigned __int8 *)a1 == v14 )
    v15 = sub_ACADE0(*((__int64 ***)v14 + 1));
  if ( v18 != (unsigned __int8 **)v20 )
    _libc_free(v18, v13);
  return v15;
}
