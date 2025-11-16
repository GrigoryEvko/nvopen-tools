// Function: sub_D4D010
// Address: 0xd4d010
//
__int64 __fastcall sub_D4D010(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 *v4; // r14
  __int64 *v5; // rbx
  __int64 i; // rax
  __int64 v7; // rdi
  unsigned int v8; // ecx
  __int64 *v9; // rbx
  __int64 *v10; // r13
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi

  v3 = a1 + 176;
  *(_QWORD *)(v3 - 176) = &unk_49DDFC8;
  sub_D4CC10(v3, a2);
  v4 = *(__int64 **)(a1 + 248);
  v5 = &v4[*(unsigned int *)(a1 + 256)];
  if ( v4 != v5 )
  {
    for ( i = *(_QWORD *)(a1 + 248); ; i = *(_QWORD *)(a1 + 248) )
    {
      v7 = *v4;
      v8 = (unsigned int)(((__int64)v4 - i) >> 3) >> 7;
      a2 = 4096LL << v8;
      if ( v8 >= 0x1E )
        a2 = 0x40000000000LL;
      ++v4;
      sub_C7D6A0(v7, a2, 16);
      if ( v5 == v4 )
        break;
    }
  }
  v9 = *(__int64 **)(a1 + 296);
  v10 = &v9[2 * *(unsigned int *)(a1 + 304)];
  if ( v9 != v10 )
  {
    do
    {
      a2 = v9[1];
      v11 = *v9;
      v9 += 2;
      sub_C7D6A0(v11, a2, 16);
    }
    while ( v10 != v9 );
    v10 = *(__int64 **)(a1 + 296);
  }
  if ( v10 != (__int64 *)(a1 + 312) )
    _libc_free(v10, a2);
  v12 = *(_QWORD *)(a1 + 248);
  if ( v12 != a1 + 264 )
    _libc_free(v12, a2);
  v13 = *(_QWORD *)(a1 + 208);
  if ( v13 )
    j_j___libc_free_0(v13, *(_QWORD *)(a1 + 224) - v13);
  sub_C7D6A0(*(_QWORD *)(a1 + 184), 16LL * *(unsigned int *)(a1 + 200), 8);
  *(_QWORD *)a1 = &unk_49DAF80;
  sub_BB9100(a1);
  return j_j___libc_free_0(a1, 328);
}
