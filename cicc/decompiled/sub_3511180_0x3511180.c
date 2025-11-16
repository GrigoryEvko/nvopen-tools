// Function: sub_3511180
// Address: 0x3511180
//
__int64 __fastcall sub_3511180(__int64 a1)
{
  __int64 v2; // rsi
  __int64 *v3; // r14
  __int64 *v4; // rbx
  __int64 i; // rax
  __int64 v6; // rdi
  unsigned int v7; // ecx
  __int64 v8; // rsi
  __int64 *v9; // rbx
  unsigned __int64 v10; // r13
  __int64 v11; // rsi
  __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // rbx
  __int64 v16; // r13
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // r13
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi

  v2 = *(unsigned int *)(a1 + 912);
  *(_QWORD *)a1 = off_4A389F0;
  sub_C7D6A0(*(_QWORD *)(a1 + 896), 16 * v2, 8);
  sub_3510940(a1 + 792);
  v3 = *(__int64 **)(a1 + 808);
  v4 = &v3[*(unsigned int *)(a1 + 816)];
  if ( v3 != v4 )
  {
    for ( i = *(_QWORD *)(a1 + 808); ; i = *(_QWORD *)(a1 + 808) )
    {
      v6 = *v3;
      v7 = (unsigned int)(((__int64)v3 - i) >> 3) >> 7;
      v8 = 4096LL << v7;
      if ( v7 >= 0x1E )
        v8 = 0x40000000000LL;
      ++v3;
      sub_C7D6A0(v6, v8, 16);
      if ( v4 == v3 )
        break;
    }
  }
  v9 = *(__int64 **)(a1 + 856);
  v10 = (unsigned __int64)&v9[2 * *(unsigned int *)(a1 + 864)];
  if ( v9 != (__int64 *)v10 )
  {
    do
    {
      v11 = v9[1];
      v12 = *v9;
      v9 += 2;
      sub_C7D6A0(v12, v11, 16);
    }
    while ( (__int64 *)v10 != v9 );
    v10 = *(_QWORD *)(a1 + 856);
  }
  if ( v10 != a1 + 872 )
    _libc_free(v10);
  v13 = *(_QWORD *)(a1 + 808);
  if ( v13 != a1 + 824 )
    _libc_free(v13);
  v14 = *(unsigned int *)(a1 + 768);
  if ( (_DWORD)v14 )
  {
    v15 = *(_QWORD *)(a1 + 752);
    v16 = v15 + 32 * v14;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v15 <= 0xFFFFFFFD )
        {
          v17 = *(_QWORD *)(v15 + 8);
          if ( v17 )
            break;
        }
        v15 += 32;
        if ( v16 == v15 )
          goto LABEL_20;
      }
      v15 += 32;
      j_j___libc_free_0(v17);
    }
    while ( v16 != v15 );
LABEL_20:
    v14 = *(unsigned int *)(a1 + 768);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 752), 32 * v14, 8);
  v18 = *(_QWORD *)(a1 + 664);
  if ( v18 != a1 + 680 )
    _libc_free(v18);
  v19 = *(_QWORD *)(a1 + 536);
  if ( v19 )
  {
    sub_C7D6A0(*(_QWORD *)(v19 + 16), 16LL * *(unsigned int *)(v19 + 32), 8);
    j_j___libc_free_0(v19);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 496), 24LL * *(unsigned int *)(a1 + 512), 8);
  v20 = *(_QWORD *)(a1 + 344);
  if ( v20 != a1 + 360 )
    _libc_free(v20);
  v21 = *(_QWORD *)(a1 + 200);
  if ( v21 != a1 + 216 )
    _libc_free(v21);
  *(_QWORD *)a1 = &unk_49DAF80;
  return sub_BB9100(a1);
}
