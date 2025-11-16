// Function: sub_23B8B10
// Address: 0x23b8b10
//
void __fastcall sub_23B8B10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  int v9; // edx
  __int64 v10; // rax
  __int64 v11; // rax
  int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // rax
  int v15; // edx
  __int64 *v16; // rax
  __m128i *v17; // r15
  __m128i *v18; // rax
  unsigned __int64 v19; // rdi
  int v20; // eax
  __m128i *v21; // r15
  __m128i *v22; // rax
  unsigned __int64 v23; // rdi
  int v24; // eax
  __m128i *v25; // r15
  __m128i *v26; // rax
  unsigned __int64 v27; // rdi
  int v28; // r12d
  int v29; // [rsp-50h] [rbp-50h]
  int v30; // [rsp-50h] [rbp-50h]
  unsigned __int64 v31[8]; // [rsp-40h] [rbp-40h] BYREF

  if ( (_BYTE)qword_4FDF2E8 )
  {
    v8 = *(unsigned int *)(a2 + 296);
    v9 = v8;
    if ( *(_DWORD *)(a2 + 300) <= (unsigned int)v8 )
    {
      v17 = (__m128i *)sub_C8D7D0(a2 + 288, a2 + 304, 0, 0x20u, v31, a6);
      v18 = &v17[2 * *(unsigned int *)(a2 + 296)];
      if ( v18 )
      {
        v18->m128i_i64[0] = a1;
        v18->m128i_i64[1] = a3;
        v18[1].m128i_i8[0] = 0;
        v18[1].m128i_i64[1] = (__int64)off_4CDFC10 + 2;
      }
      sub_BC3E80(a2 + 288, v17);
      v19 = *(_QWORD *)(a2 + 288);
      v20 = v31[0];
      if ( a2 + 304 != v19 )
      {
        v29 = v31[0];
        _libc_free(v19);
        v20 = v29;
      }
      ++*(_DWORD *)(a2 + 296);
      *(_QWORD *)(a2 + 288) = v17;
      *(_DWORD *)(a2 + 300) = v20;
    }
    else
    {
      v10 = *(_QWORD *)(a2 + 288) + 32 * v8;
      if ( v10 )
      {
        *(_QWORD *)v10 = a1;
        *(_QWORD *)(v10 + 8) = a3;
        *(_BYTE *)(v10 + 16) = 0;
        *(_QWORD *)(v10 + 24) = (char *)off_4CDFC10 + 2;
        v9 = *(_DWORD *)(a2 + 296);
      }
      *(_DWORD *)(a2 + 296) = v9 + 1;
    }
    v11 = *(unsigned int *)(a2 + 584);
    v12 = v11;
    if ( *(_DWORD *)(a2 + 588) <= (unsigned int)v11 )
    {
      v21 = (__m128i *)sub_C8D7D0(a2 + 576, a2 + 592, 0, 0x20u, v31, a6);
      v22 = &v21[2 * *(unsigned int *)(a2 + 584)];
      if ( v22 )
      {
        v22->m128i_i64[0] = a1;
        v22[1].m128i_i64[1] = (__int64)&off_4CDFC38 + 2;
      }
      sub_BC4100(a2 + 576, v21);
      v23 = *(_QWORD *)(a2 + 576);
      v24 = v31[0];
      if ( a2 + 592 != v23 )
      {
        v30 = v31[0];
        _libc_free(v23);
        v24 = v30;
      }
      ++*(_DWORD *)(a2 + 584);
      *(_QWORD *)(a2 + 576) = v21;
      *(_DWORD *)(a2 + 588) = v24;
    }
    else
    {
      v13 = (__int64 *)(*(_QWORD *)(a2 + 576) + 32 * v11);
      if ( v13 )
      {
        *v13 = a1;
        v13[3] = (__int64)&off_4CDFC38 + 2;
        v12 = *(_DWORD *)(a2 + 584);
      }
      *(_DWORD *)(a2 + 584) = v12 + 1;
    }
    v14 = *(unsigned int *)(a2 + 440);
    v15 = v14;
    if ( *(_DWORD *)(a2 + 444) <= (unsigned int)v14 )
    {
      v25 = (__m128i *)sub_C8D7D0(a2 + 432, a2 + 448, 0, 0x20u, v31, a6);
      v26 = &v25[2 * *(unsigned int *)(a2 + 440)];
      if ( v26 )
      {
        v26->m128i_i64[0] = a1;
        v26->m128i_i64[1] = a3;
        v26[1].m128i_i64[1] = (__int64)off_4CDFC30 + 2;
      }
      sub_BC3FC0(a2 + 432, v25);
      v27 = *(_QWORD *)(a2 + 432);
      v28 = v31[0];
      if ( a2 + 448 != v27 )
        _libc_free(v27);
      ++*(_DWORD *)(a2 + 440);
      *(_QWORD *)(a2 + 432) = v25;
      *(_DWORD *)(a2 + 444) = v28;
    }
    else
    {
      v16 = (__int64 *)(*(_QWORD *)(a2 + 432) + 32 * v14);
      if ( v16 )
      {
        *v16 = a1;
        v16[1] = a3;
        v16[3] = (__int64)off_4CDFC30 + 2;
        v15 = *(_DWORD *)(a2 + 440);
      }
      *(_DWORD *)(a2 + 440) = v15 + 1;
    }
  }
}
