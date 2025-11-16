// Function: sub_370BFD0
// Address: 0x370bfd0
//
unsigned __int64 *__fastcall sub_370BFD0(unsigned __int64 *a1, _QWORD *a2, unsigned int *a3, const __m128i *a4)
{
  __int64 v6; // rdi
  __int64 v7; // r14
  int v9; // r8d
  unsigned int v10; // eax
  unsigned __int32 v11; // edx
  __int64 v12; // rax
  __int64 v13; // rsi
  unsigned int v14; // ebx
  __int64 (*v15)(void); // rax
  int v16; // r8d
  unsigned __int32 v17; // eax
  unsigned __int64 v19; // [rsp+18h] [rbp-58h] BYREF
  _OWORD v20[2]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v21; // [rsp+40h] [rbp-30h]

  v6 = a2[7];
  v7 = a2[5];
  if ( v6 )
  {
    if ( !v7 && !a2[6] )
    {
      if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v6 + 40LL))(v6) )
      {
        v12 = a4[2].m128i_i64[0];
        v20[0] = _mm_loadu_si128(a4);
        v21 = v12;
        v20[1] = _mm_loadu_si128(a4 + 1);
        if ( (unsigned __int8)v12 > 1u )
          (*(void (__fastcall **)(_QWORD, _OWORD *))(*(_QWORD *)a2[7] + 24LL))(a2[7], v20);
      }
      (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(*(_QWORD *)a2[7] + 8LL))(a2[7], (int)*a3, 4);
      if ( a2[7] && !a2[5] && !a2[6] )
        a2[8] += 4LL;
      *a1 = 1;
      *(_QWORD *)&v20[0] = 0;
      sub_9C66B0((__int64 *)v20);
      return a1;
    }
  }
  else
  {
    v13 = a2[6];
    if ( v13 && !v7 )
    {
      v14 = *a3;
      v15 = *(__int64 (**)(void))(**(_QWORD **)(v13 + 24) + 16LL);
      if ( v15 != sub_3700C70 )
      {
        v16 = v15();
        v17 = _byteswap_ulong(v14);
        if ( v16 != 1 )
          v14 = v17;
      }
      LODWORD(v20[0]) = v14;
      sub_3719260(a1, v13, v20, 4);
      return a1;
    }
  }
  v20[0] = 0u;
  sub_1254950(&v19, v7, (__int64)v20, 4u);
  if ( (v19 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v19 & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  v9 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v7 + 24) + 16LL))(*(_QWORD *)(v7 + 24));
  v10 = **(_DWORD **)&v20[0];
  v11 = _byteswap_ulong(**(_DWORD **)&v20[0]);
  if ( v9 != 1 )
    v10 = v11;
  *a3 = v10;
  *a1 = 1;
  return a1;
}
