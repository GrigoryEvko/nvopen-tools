// Function: sub_F323D0
// Address: 0xf323d0
//
__int64 __fastcall sub_F323D0(__int64 *a1, __int64 *a2, int a3)
{
  volatile signed __int32 *v4; // r12
  signed __int32 v5; // eax
  signed __int32 v6; // eax
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 *v9; // r12
  __int64 *v10; // rbx
  __int64 *v11; // rdi
  __int64 *v12; // r15
  __int64 v13; // r14
  __int64 v14; // rdi
  __int64 v15; // r15
  _QWORD *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r12
  __int64 v20; // r13
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 v23; // rax
  const __m128i *v24; // r14
  __int64 v25; // rax
  __m128i v26; // xmm0
  __int64 v27; // rdx
  __int64 v28; // rsi
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  int v33; // eax
  __int64 v34; // [rsp+0h] [rbp-50h]
  __int64 v35; // [rsp+0h] [rbp-50h]
  int v36; // [rsp+0h] [rbp-50h]
  __int64 v37; // [rsp+8h] [rbp-48h]
  unsigned int v38; // [rsp+8h] [rbp-48h]
  unsigned int v39; // [rsp+8h] [rbp-48h]
  unsigned __int64 v40[7]; // [rsp+18h] [rbp-38h] BYREF

  switch ( a3 )
  {
    case 1:
      *a1 = *a2;
      break;
    case 2:
      v15 = *a2;
      v16 = (_QWORD *)sub_22077B0(104);
      v19 = (__int64)v16;
      if ( v16 )
      {
        v20 = (__int64)(v16 + 2);
        *v16 = v16 + 2;
        v16[1] = 0x100000000LL;
        v21 = *(unsigned int *)(v15 + 8);
        if ( (_DWORD)v21 && v16 != (_QWORD *)v15 )
        {
          v23 = 1;
          if ( (_DWORD)v21 != 1 )
          {
            v39 = *(_DWORD *)(v15 + 8);
            v28 = sub_C8D7D0(v19, v20, (unsigned int)v21, 0x48u, v40, v21);
            sub_F32260(v19, v28, v29, v30, v31, v32);
            v33 = v40[0];
            v21 = v39;
            if ( v20 != *(_QWORD *)v19 )
            {
              v36 = v40[0];
              _libc_free(*(_QWORD *)v19, v28);
              v33 = v36;
              v21 = v39;
            }
            *(_QWORD *)v19 = v28;
            v20 = v28;
            *(_DWORD *)(v19 + 12) = v33;
            v23 = *(unsigned int *)(v15 + 8);
          }
          v24 = *(const __m128i **)v15;
          v25 = *(_QWORD *)v15 + 72 * v23;
          if ( *(_QWORD *)v15 != v25 )
          {
            do
            {
              if ( v20 )
              {
                v26 = _mm_loadu_si128(v24);
                *(_DWORD *)(v20 + 24) = 0;
                *(_QWORD *)(v20 + 16) = v20 + 32;
                *(_DWORD *)(v20 + 28) = 1;
                *(__m128i *)v20 = v26;
                v27 = v24[1].m128i_u32[2];
                if ( (_DWORD)v27 )
                {
                  v35 = v25;
                  v38 = v21;
                  sub_F317D0(v20 + 16, v24[1].m128i_i64, v27, v17, v18, v21);
                  v25 = v35;
                  v21 = v38;
                }
              }
              v24 = (const __m128i *)((char *)v24 + 72);
              v20 += 72;
            }
            while ( (const __m128i *)v25 != v24 );
          }
          *(_DWORD *)(v19 + 8) = v21;
        }
        *(_QWORD *)(v19 + 88) = *(_QWORD *)(v15 + 88);
        v22 = *(_QWORD *)(v15 + 96);
        *(_QWORD *)(v19 + 96) = v22;
        if ( v22 )
        {
          if ( &_pthread_key_create )
            _InterlockedAdd((volatile signed __int32 *)(v22 + 8), 1u);
          else
            ++*(_DWORD *)(v22 + 8);
        }
      }
      *a1 = v19;
      break;
    case 3:
      v34 = *a1;
      if ( *a1 )
      {
        v4 = *(volatile signed __int32 **)(*a1 + 96);
        if ( v4 )
        {
          if ( &_pthread_key_create )
          {
            v5 = _InterlockedExchangeAdd(v4 + 2, 0xFFFFFFFF);
          }
          else
          {
            v5 = *((_DWORD *)v4 + 2);
            *((_DWORD *)v4 + 2) = v5 - 1;
          }
          if ( v5 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v4 + 16LL))(v4);
            if ( &_pthread_key_create )
            {
              v6 = _InterlockedExchangeAdd(v4 + 3, 0xFFFFFFFF);
            }
            else
            {
              v6 = *((_DWORD *)v4 + 3);
              *((_DWORD *)v4 + 3) = v6 - 1;
            }
            if ( v6 == 1 )
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v4 + 24LL))(v4);
          }
        }
        v37 = *(_QWORD *)v34;
        v7 = *(_QWORD *)v34 + 72LL * *(unsigned int *)(v34 + 8);
        if ( *(_QWORD *)v34 != v7 )
        {
          do
          {
            v8 = *(unsigned int *)(v7 - 48);
            v9 = *(__int64 **)(v7 - 56);
            v7 -= 72;
            v10 = &v9[5 * v8];
            if ( v9 != v10 )
            {
              do
              {
                v10 -= 5;
                v11 = (__int64 *)v10[2];
                if ( v11 != v10 + 5 )
                  _libc_free(v11, a2);
                v12 = (__int64 *)*v10;
                v13 = *v10 + 80LL * *((unsigned int *)v10 + 2);
                if ( *v10 != v13 )
                {
                  do
                  {
                    v13 -= 80;
                    v14 = *(_QWORD *)(v13 + 8);
                    if ( v14 != v13 + 24 )
                      _libc_free(v14, a2);
                  }
                  while ( v12 != (__int64 *)v13 );
                  v12 = (__int64 *)*v10;
                }
                if ( v12 != v10 + 2 )
                  _libc_free(v12, a2);
              }
              while ( v9 != v10 );
              v9 = *(__int64 **)(v7 + 16);
            }
            if ( v9 != (__int64 *)(v7 + 32) )
              _libc_free(v9, a2);
          }
          while ( v37 != v7 );
          v7 = *(_QWORD *)v34;
        }
        if ( v7 != v34 + 16 )
          _libc_free(v7, a2);
        j_j___libc_free_0(v34, 104);
      }
      break;
  }
  return 0;
}
