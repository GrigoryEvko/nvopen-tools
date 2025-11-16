// Function: sub_258FAF0
// Address: 0x258faf0
//
__int64 __fastcall sub_258FAF0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  int v8; // edi
  signed int v9; // r15d
  __int64 v10; // rdx
  unsigned int v11; // r12d
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // r9
  __m128i v16; // [rsp+0h] [rbp-80h] BYREF
  __int64 v17; // [rsp+10h] [rbp-70h]
  char *v18; // [rsp+18h] [rbp-68h] BYREF
  __int64 v19; // [rsp+20h] [rbp-60h]
  char v20; // [rsp+28h] [rbp-58h] BYREF
  __int64 v21; // [rsp+30h] [rbp-50h] BYREF
  _BYTE *v22; // [rsp+38h] [rbp-48h] BYREF
  __int64 v23; // [rsp+40h] [rbp-40h]
  _BYTE v24[56]; // [rsp+48h] [rbp-38h] BYREF

  v7 = *a2;
  v8 = *((_DWORD *)a2 + 4);
  v18 = &v20;
  v19 = 0;
  v17 = v7;
  if ( v8 )
  {
    sub_2538240((__int64)&v18, (char **)a2 + 1, a3, a4, a5, a6);
    v9 = **(_DWORD **)a1;
    v22 = v24;
    v23 = 0;
    v21 = v17;
    if ( (_DWORD)v19 )
      sub_2538550((__int64)&v22, (__int64)&v18, v13, (unsigned int)v19, v14, v15);
  }
  else
  {
    v9 = **(_DWORD **)a1;
    v21 = v7;
    v22 = v24;
    v23 = 0;
  }
  v16.m128i_i64[0] = sub_254CA10((__int64)&v21, v9);
  v16.m128i_i64[1] = v10;
  if ( v22 != v24 )
    _libc_free((unsigned __int64)v22);
  v11 = 0;
  if ( (unsigned __int8)sub_2509800(&v16) )
    v11 = sub_258F340(*(_QWORD **)(a1 + 8), *(_QWORD *)(a1 + 16), &v16, 0, &v21, 0, 0);
  if ( v18 != &v20 )
    _libc_free((unsigned __int64)v18);
  return v11;
}
