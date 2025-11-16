// Function: sub_30B99C0
// Address: 0x30b99c0
//
__int64 __fastcall sub_30B99C0(__int64 a1)
{
  __int64 v1; // r13
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 result; // rax
  _QWORD *v6; // rdx
  __int64 *v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rbx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 *v12; // rax
  char v13; // dl
  __int64 v14; // r15
  __int64 v15; // r9
  __int64 *v16; // rdx
  __int64 v17; // rdi
  __int64 *v18; // rax
  const __m128i *v19; // rax
  unsigned __int64 v20; // rdi
  __m128i *v21; // rdx
  int v22; // ebx
  int v23; // eax
  __int64 v24; // [rsp+8h] [rbp-48h]
  unsigned __int64 v25[7]; // [rsp+18h] [rbp-38h] BYREF

  v1 = a1 + 112;
  v3 = *(_QWORD *)(a1 + 96);
  v4 = *(unsigned int *)(a1 + 104);
  while ( 1 )
  {
    result = v3 + 40 * v4 - 40;
    v6 = *(_QWORD **)(result + 16);
    if ( *(_QWORD **)result == v6 )
      return result;
    *(_QWORD *)(result + 16) = v6 + 1;
    v9 = (*(__int64 (__fastcall **)(_QWORD))(result + 24))(*v6);
    if ( !*(_BYTE *)(a1 + 28) )
      goto LABEL_10;
    v12 = *(__int64 **)(a1 + 8);
    v8 = *(unsigned int *)(a1 + 20);
    v7 = &v12[v8];
    if ( v12 == v7 )
    {
LABEL_15:
      if ( (unsigned int)v8 < *(_DWORD *)(a1 + 16) )
      {
        *(_DWORD *)(a1 + 20) = v8 + 1;
        *v7 = v9;
        LODWORD(v4) = *(_DWORD *)(a1 + 104);
        ++*(_QWORD *)a1;
        goto LABEL_11;
      }
LABEL_10:
      sub_C8CC70(a1, v9, (__int64)v7, v8, v10, v11);
      v4 = *(unsigned int *)(a1 + 104);
      if ( !v13 )
        goto LABEL_9;
LABEL_11:
      v14 = *(_QWORD *)(v9 + 40);
      v15 = v14 + 8LL * *(unsigned int *)(v9 + 48);
      if ( *(_DWORD *)(a1 + 108) <= (unsigned int)v4 )
      {
        v24 = v14 + 8LL * *(unsigned int *)(v9 + 48);
        v3 = sub_C8D7D0(a1 + 96, v1, 0, 0x28u, v25, v15);
        v17 = 5LL * *(unsigned int *)(a1 + 104);
        v18 = (__int64 *)(v17 * 8 + v3);
        if ( v17 * 8 + v3 )
        {
          v18[2] = v14;
          v18[4] = v9;
          *v18 = v24;
          v18[1] = (__int64)sub_30B9540;
          v18[3] = (__int64)sub_30B9540;
          v17 = 5LL * *(unsigned int *)(a1 + 104);
        }
        v19 = *(const __m128i **)(a1 + 96);
        v20 = (unsigned __int64)&v19->m128i_u64[v17];
        if ( v19 != (const __m128i *)v20 )
        {
          v21 = (__m128i *)v3;
          do
          {
            if ( v21 )
            {
              *v21 = _mm_loadu_si128(v19);
              v21[1] = _mm_loadu_si128(v19 + 1);
              v21[2].m128i_i64[0] = v19[2].m128i_i64[0];
            }
            v19 = (const __m128i *)((char *)v19 + 40);
            v21 = (__m128i *)((char *)v21 + 40);
          }
          while ( (const __m128i *)v20 != v19 );
          v20 = *(_QWORD *)(a1 + 96);
        }
        v22 = v25[0];
        if ( v1 != v20 )
          _libc_free(v20);
        v23 = *(_DWORD *)(a1 + 104);
        *(_QWORD *)(a1 + 96) = v3;
        *(_DWORD *)(a1 + 108) = v22;
        v4 = (unsigned int)(v23 + 1);
        *(_DWORD *)(a1 + 104) = v4;
      }
      else
      {
        v3 = *(_QWORD *)(a1 + 96);
        v16 = (__int64 *)(v3 + 40LL * (unsigned int)v4);
        if ( v16 )
        {
          *v16 = v15;
          v16[2] = v14;
          v16[1] = (__int64)sub_30B9540;
          v16[3] = (__int64)sub_30B9540;
          v16[4] = v9;
          LODWORD(v4) = *(_DWORD *)(a1 + 104);
          v3 = *(_QWORD *)(a1 + 96);
        }
        v4 = (unsigned int)(v4 + 1);
        *(_DWORD *)(a1 + 104) = v4;
      }
    }
    else
    {
      while ( v9 != *v12 )
      {
        if ( v7 == ++v12 )
          goto LABEL_15;
      }
      v4 = *(unsigned int *)(a1 + 104);
LABEL_9:
      v3 = *(_QWORD *)(a1 + 96);
    }
  }
}
