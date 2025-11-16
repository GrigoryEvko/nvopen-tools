// Function: sub_30712E0
// Address: 0x30712e0
//
__int64 __fastcall sub_30712E0(__int64 a1)
{
  __int64 v2; // rsi
  __int64 v3; // rsi
  __int64 *v4; // r14
  __int64 *v5; // rbx
  __int64 i; // rax
  __int64 v7; // rdi
  unsigned int v8; // ecx
  __int64 *v9; // rbx
  unsigned __int64 v10; // r13
  __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  __int64 v13; // rdi

  v2 = *(unsigned int *)(a1 + 539440);
  *(_QWORD *)a1 = &unk_4A30900;
  v3 = 16 * v2;
  sub_C7D6A0(*(_QWORD *)(a1 + 539424), v3, 8);
  v4 = *(__int64 **)(a1 + 539328);
  v5 = &v4[*(unsigned int *)(a1 + 539336)];
  if ( v4 != v5 )
  {
    for ( i = *(_QWORD *)(a1 + 539328); ; i = *(_QWORD *)(a1 + 539328) )
    {
      v7 = *v4;
      v8 = (unsigned int)(((__int64)v4 - i) >> 3) >> 7;
      v3 = 4096LL << v8;
      if ( v8 >= 0x1E )
        v3 = 0x40000000000LL;
      ++v4;
      sub_C7D6A0(v7, v3, 16);
      if ( v5 == v4 )
        break;
    }
  }
  v9 = *(__int64 **)(a1 + 539376);
  v10 = (unsigned __int64)&v9[2 * *(unsigned int *)(a1 + 539384)];
  if ( v9 != (__int64 *)v10 )
  {
    do
    {
      v3 = v9[1];
      v11 = *v9;
      v9 += 2;
      sub_C7D6A0(v11, v3, 16);
    }
    while ( (__int64 *)v10 != v9 );
    v10 = *(_QWORD *)(a1 + 539376);
  }
  if ( v10 != a1 + 539392 )
    _libc_free(v10);
  v12 = *(_QWORD *)(a1 + 539328);
  if ( v12 != a1 + 539344 )
    _libc_free(v12);
  sub_3059A40(a1 + 1288);
  v13 = *(_QWORD *)(a1 + 1272);
  if ( v13 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v13 + 8LL))(v13);
  *(_QWORD *)a1 = &unk_4A379F8;
  return sub_23CE6A0(a1, v3);
}
