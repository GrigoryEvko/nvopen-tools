// Function: sub_254DEA0
// Address: 0x254dea0
//
__int64 __fastcall sub_254DEA0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  int v8; // edi
  signed int v9; // r14d
  __int64 v10; // rdx
  unsigned int v11; // r13d
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __m128i v17; // [rsp+0h] [rbp-80h] BYREF
  __int64 v18; // [rsp+10h] [rbp-70h]
  char *v19; // [rsp+18h] [rbp-68h] BYREF
  __int64 v20; // [rsp+20h] [rbp-60h]
  char v21; // [rsp+28h] [rbp-58h] BYREF
  __int64 v22; // [rsp+30h] [rbp-50h] BYREF
  _BYTE *v23; // [rsp+38h] [rbp-48h] BYREF
  __int64 v24; // [rsp+40h] [rbp-40h]
  _BYTE v25[56]; // [rsp+48h] [rbp-38h] BYREF

  v7 = *a2;
  v8 = *((_DWORD *)a2 + 4);
  v19 = &v21;
  v20 = 0;
  v18 = v7;
  if ( v8 )
  {
    sub_2538240((__int64)&v19, (char **)a2 + 1, a3, a4, a5, a6);
    v9 = **(_DWORD **)a1;
    v23 = v25;
    v24 = 0;
    v22 = v18;
    if ( (_DWORD)v20 )
      sub_2538550((__int64)&v23, (__int64)&v19, v13, v14, v15, v16);
  }
  else
  {
    v9 = **(_DWORD **)a1;
    v22 = v7;
    v23 = v25;
    v24 = 0;
  }
  v17.m128i_i64[0] = sub_254CA10((__int64)&v22, v9);
  v17.m128i_i64[1] = v10;
  if ( v23 != v25 )
    _libc_free((unsigned __int64)v23);
  if ( (unsigned __int8)sub_2509800(&v17)
    && (unsigned __int8)sub_2526B50(
                          *(_QWORD *)(a1 + 8),
                          &v17,
                          *(_QWORD *)(a1 + 16),
                          *(_QWORD *)(a1 + 24),
                          2u,
                          *(_BYTE **)(a1 + 32),
                          1u) )
  {
    v11 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 16) + 105LL);
  }
  else
  {
    v11 = 0;
  }
  if ( v19 != &v21 )
    _libc_free((unsigned __int64)v19);
  return v11;
}
