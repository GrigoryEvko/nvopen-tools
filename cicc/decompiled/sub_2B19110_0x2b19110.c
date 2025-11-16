// Function: sub_2B19110
// Address: 0x2b19110
//
__int64 __fastcall sub_2B19110(__int64 a1, __int64 a2)
{
  unsigned int v3; // r13d
  unsigned int v4; // edx
  __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 *v7; // rdi
  __int64 result; // rax
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  __int64 v11; // rbx
  unsigned __int8 v12; // [rsp+Fh] [rbp-D1h]
  unsigned __int64 v13; // [rsp+10h] [rbp-D0h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-C8h]
  __m128i v15; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v16; // [rsp+30h] [rbp-B0h]
  __int64 v17; // [rsp+38h] [rbp-A8h]
  __int64 v18; // [rsp+40h] [rbp-A0h]
  __int64 v19; // [rsp+48h] [rbp-98h]
  __int64 v20; // [rsp+50h] [rbp-90h]
  __int64 v21; // [rsp+58h] [rbp-88h]
  __int16 v22; // [rsp+60h] [rbp-80h]
  __m128i v23; // [rsp+70h] [rbp-70h] BYREF
  __int64 v24; // [rsp+80h] [rbp-60h]
  __int64 v25; // [rsp+88h] [rbp-58h]
  __int64 v26; // [rsp+90h] [rbp-50h]
  __int64 v27; // [rsp+98h] [rbp-48h]
  __int64 v28; // [rsp+A0h] [rbp-40h]
  __int64 v29; // [rsp+A8h] [rbp-38h]
  __int16 v30; // [rsp+B0h] [rbp-30h]

  v3 = **(_DWORD **)(a1 + 8);
  v4 = **(_DWORD **)a1;
  v14 = v4;
  if ( v4 > 0x40 )
  {
    sub_C43690((__int64)&v13, 0, 0);
    v4 = v14;
  }
  else
  {
    v13 = 0;
  }
  if ( v3 != v4 )
  {
    if ( v3 > 0x3F || v4 > 0x40 )
      sub_C43C90(&v13, v3, v4);
    else
      v13 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v3 + 64 - (unsigned __int8)v4) << v3;
  }
  v5 = *(_QWORD *)(a1 + 16);
  v16 = 0;
  v6 = *(_QWORD *)(v5 + 3344);
  v22 = 257;
  v17 = 0;
  v15 = (__m128i)v6;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v7 = *(__int64 **)(a2 - 8);
  else
    v7 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  result = sub_9AC230(*v7, (__int64)&v13, &v15, 0);
  if ( (_BYTE)result )
  {
    v9 = *(_QWORD *)(a1 + 16);
    v24 = 0;
    v10 = *(_QWORD *)(v9 + 3344);
    v25 = 0;
    v26 = 0;
    v23 = (__m128i)v10;
    v27 = 0;
    v28 = 0;
    v29 = 0;
    v30 = 257;
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v11 = *(_QWORD *)(a2 - 8);
    else
      v11 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    result = sub_9AC230(*(_QWORD *)(v11 + 32), (__int64)&v13, &v23, 0);
  }
  if ( v14 > 0x40 )
  {
    if ( v13 )
    {
      v12 = result;
      j_j___libc_free_0_0(v13);
      return v12;
    }
  }
  return result;
}
