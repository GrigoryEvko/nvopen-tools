// Function: sub_2277160
// Address: 0x2277160
//
__int64 __fastcall sub_2277160(__int64 a1)
{
  _QWORD *v2; // rdi
  __int64 v3; // r14
  unsigned __int64 v4; // rdi
  __int64 v5; // r13
  __int64 v6; // r14
  _QWORD *v7; // rbx
  __int64 v8; // r15
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  __int64 v15; // r13
  __int64 v16; // rbx
  __int64 v17; // r13
  __int64 v18; // r15
  unsigned __int64 v19; // r14
  const __m128i *v20; // rdi
  __int64 v22; // [rsp+8h] [rbp-48h]
  __int64 v23; // [rsp+10h] [rbp-40h]
  __int64 v24; // [rsp+18h] [rbp-38h]

  v2 = (_QWORD *)(a1 + 1072);
  *v2 = &unk_4A0A978;
  sub_2272350((__int64)v2);
  sub_23B3CB0(a1 + 1016);
  sub_23B3B90(a1 + 984);
  sub_23B61B0(a1 + 936);
  sub_23B3FF0(a1 + 880);
  if ( *(_DWORD *)(a1 + 868) )
  {
    v3 = *(unsigned int *)(a1 + 864);
    v4 = *(_QWORD *)(a1 + 856);
    if ( (_DWORD)v3 )
    {
      v5 = 0;
      v24 = 8 * v3;
      do
      {
        v6 = *(_QWORD *)(v4 + v5);
        if ( v6 != -8 && v6 )
        {
          v7 = *(_QWORD **)(v6 + 24);
          v8 = *(_QWORD *)v6 + 65LL;
          while ( v7 )
          {
            v9 = (unsigned __int64)v7;
            v7 = (_QWORD *)*v7;
            j_j___libc_free_0(v9);
          }
          memset(*(void **)(v6 + 8), 0, 8LL * *(_QWORD *)(v6 + 16));
          v10 = *(_QWORD *)(v6 + 8);
          *(_QWORD *)(v6 + 32) = 0;
          *(_QWORD *)(v6 + 24) = 0;
          if ( v10 != v6 + 56 )
            j_j___libc_free_0(v10);
          sub_C7D6A0(v6, v8, 8);
          v4 = *(_QWORD *)(a1 + 856);
        }
        v5 += 8;
      }
      while ( v24 != v5 );
    }
  }
  else
  {
    v4 = *(_QWORD *)(a1 + 856);
  }
  _libc_free(v4);
  sub_23B3C60(a1 + 808);
  v11 = *(_QWORD *)(a1 + 664);
  if ( v11 != a1 + 680 )
    _libc_free(v11);
  sub_BC3B50(a1 + 216);
  v12 = *(_QWORD *)(a1 + 544);
  if ( v12 != a1 + 560 )
    _libc_free(v12);
  v13 = *(_QWORD *)(a1 + 464);
  if ( v13 != a1 + 480 )
    _libc_free(v13);
  v14 = *(_QWORD *)(a1 + 440);
  if ( *(_DWORD *)(a1 + 452) )
  {
    v15 = *(unsigned int *)(a1 + 448);
    if ( (_DWORD)v15 )
    {
      v16 = 0;
      v23 = 8 * v15;
      do
      {
        v17 = *(_QWORD *)(v14 + v16);
        if ( v17 != -8 && v17 )
        {
          v18 = *(_QWORD *)(v17 + 8);
          v19 = v18 + 8LL * *(unsigned int *)(v17 + 16);
          v22 = *(_QWORD *)v17 + 57LL;
          if ( v18 != v19 )
          {
            do
            {
              v20 = *(const __m128i **)(v19 - 8);
              v19 -= 8LL;
              if ( v20 )
              {
                sub_C9F8C0(v20);
                j_j___libc_free_0((unsigned __int64)v20);
              }
            }
            while ( v18 != v19 );
            v19 = *(_QWORD *)(v17 + 8);
          }
          if ( v19 != v17 + 24 )
            _libc_free(v19);
          sub_C7D6A0(v17, v22, 8);
          v14 = *(_QWORD *)(a1 + 440);
        }
        v16 += 8;
      }
      while ( v23 != v16 );
    }
  }
  _libc_free(v14);
  sub_C9F930(a1 + 328);
  sub_C9F930(a1 + 216);
  return sub_23B2BA0(a1);
}
