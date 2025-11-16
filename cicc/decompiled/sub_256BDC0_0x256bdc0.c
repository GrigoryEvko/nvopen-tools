// Function: sub_256BDC0
// Address: 0x256bdc0
//
unsigned __int64 __fastcall sub_256BDC0(__int64 *a1, __int64 a2)
{
  const __m128i *v2; // r12
  unsigned __int64 result; // rax
  __int64 v4; // rbx
  unsigned int v5; // esi
  __int64 v6; // r10
  unsigned int v7; // r13d
  __int64 *v8; // rdi
  __int64 v9; // r9
  __int64 v10; // r8
  int v11; // r15d
  __int64 *v12; // rax
  __int64 v13; // r15
  int v14; // ecx
  __m128i v15; // xmm2
  unsigned __int64 v16; // rdx
  unsigned int *v17; // r13
  unsigned int v18; // ebx
  unsigned __int64 v19; // rdx
  unsigned int *v20; // r12
  char v21; // r14
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  int v25; // ecx
  const __m128i *v26; // [rsp+8h] [rbp-88h]
  __int64 v27; // [rsp+10h] [rbp-80h]
  _QWORD *v28; // [rsp+18h] [rbp-78h]
  __int64 v29; // [rsp+20h] [rbp-70h]
  const __m128i *v30; // [rsp+28h] [rbp-68h]
  unsigned int *v31; // [rsp+30h] [rbp-60h]
  __int64 *v33; // [rsp+48h] [rbp-48h] BYREF
  __m128i v34; // [rsp+50h] [rbp-40h] BYREF

  v2 = *(const __m128i **)a2;
  result = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
  v30 = (const __m128i *)result;
  if ( result != *(_QWORD *)a2 )
  {
    do
    {
      v4 = *a1;
      v34 = _mm_loadu_si128(v2);
      v5 = *(_DWORD *)(v4 + 160);
      if ( !v5 )
      {
        ++*(_QWORD *)(v4 + 136);
        v33 = 0;
        goto LABEL_16;
      }
      v6 = *(_QWORD *)(v4 + 144);
      v7 = (v5 - 1)
         & (((0xBF58476D1CE4E5B9LL
            * ((unsigned int)(37 * v34.m128i_i32[2]) | ((unsigned __int64)(unsigned int)(37 * v34.m128i_i32[0]) << 32))) >> 31)
          ^ (756364221 * v34.m128i_i32[2]));
      v8 = (__int64 *)(v6 + 96LL * v7);
      v9 = v8[1];
      v10 = *v8;
      if ( *(_OWORD *)&v34 != *(_OWORD *)v8 )
      {
        v11 = 1;
        v12 = 0;
        while ( 1 )
        {
          if ( v10 == 0x7FFFFFFFFFFFFFFFLL )
          {
            if ( v9 == 0x7FFFFFFFFFFFFFFFLL )
            {
              v25 = *(_DWORD *)(v4 + 152);
              if ( !v12 )
                v12 = v8;
              ++*(_QWORD *)(v4 + 136);
              v14 = v25 + 1;
              v33 = v12;
              if ( 4 * v14 < 3 * v5 )
              {
                if ( v5 - *(_DWORD *)(v4 + 156) - v14 <= v5 >> 3 )
                {
LABEL_17:
                  sub_2569C80(v4 + 136, v5);
                  sub_255DB00(v4 + 136, v34.m128i_i64, &v33);
                  v14 = *(_DWORD *)(v4 + 152) + 1;
                  v12 = v33;
                }
                *(_DWORD *)(v4 + 152) = v14;
                if ( *v12 != 0x7FFFFFFFFFFFFFFFLL || v12[1] != 0x7FFFFFFFFFFFFFFFLL )
                  --*(_DWORD *)(v4 + 156);
                v15 = _mm_loadu_si128(&v34);
                v12[2] = (__int64)(v12 + 4);
                v13 = (__int64)(v12 + 2);
                *(__m128i *)v12 = v15;
                *((_OWORD *)v12 + 3) = 0;
                v12[3] = 0x400000000LL;
                *((_DWORD *)v12 + 14) = 0;
                v12[8] = 0;
                v12[9] = (__int64)(v12 + 7);
                v12[10] = (__int64)(v12 + 7);
                v12[11] = 0;
                *((_OWORD *)v12 + 2) = 0;
                v31 = (unsigned int *)a1[1];
LABEL_21:
                v16 = *(unsigned int *)(v13 + 8);
                v17 = (unsigned int *)(*(_QWORD *)v13 + 4 * v16);
                if ( *(unsigned int **)v13 == v17 )
                {
                  if ( v16 <= 3 )
                  {
                    v18 = *v31;
                    goto LABEL_27;
                  }
                  v27 = v13 + 32;
                }
                else
                {
                  v18 = *v31;
                  result = *(_QWORD *)v13;
                  while ( *(_DWORD *)result != v18 )
                  {
                    result += 4LL;
                    if ( v17 == (unsigned int *)result )
                      goto LABEL_26;
                  }
                  if ( v17 != (unsigned int *)result )
                    goto LABEL_13;
LABEL_26:
                  if ( v16 <= 3 )
                  {
LABEL_27:
                    result = *(unsigned int *)(v13 + 12);
                    v19 = v16 + 1;
                    if ( v19 > result )
                    {
                      sub_C8D5F0(v13, (const void *)(v13 + 16), v19, 4u, v10, *(_QWORD *)v13);
                      result = *(_QWORD *)v13;
                      v17 = (unsigned int *)(*(_QWORD *)v13 + 4LL * *(unsigned int *)(v13 + 8));
                    }
                    *v17 = v18;
                    ++*(_DWORD *)(v13 + 8);
                    goto LABEL_13;
                  }
                  v29 = *(_QWORD *)v13 + 4 * v16;
                  v27 = v13 + 32;
                  v26 = v2;
                  v20 = *(unsigned int **)v13;
                  do
                  {
                    v23 = sub_B9AB10((_QWORD *)(v13 + 32), v13 + 40, v20);
                    if ( v24 )
                    {
                      v21 = v23 || v24 == v13 + 40 || *v20 < *(_DWORD *)(v24 + 32);
                      v28 = (_QWORD *)v24;
                      v22 = sub_22077B0(0x28u);
                      *(_DWORD *)(v22 + 32) = *v20;
                      sub_220F040(v21, v22, v28, (_QWORD *)(v13 + 40));
                      ++*(_QWORD *)(v13 + 72);
                    }
                    ++v20;
                  }
                  while ( (unsigned int *)v29 != v20 );
                  v2 = v26;
                }
                *(_DWORD *)(v13 + 8) = 0;
                result = sub_B99820(v27, v31);
                goto LABEL_13;
              }
LABEL_16:
              v5 *= 2;
              goto LABEL_17;
            }
          }
          else if ( v9 == 0x7FFFFFFFFFFFFFFELL && v10 == 0x7FFFFFFFFFFFFFFELL && !v12 )
          {
            v12 = v8;
          }
          v7 = (v5 - 1) & (v11 + v7);
          v8 = (__int64 *)(v6 + 96LL * v7);
          v10 = *v8;
          v9 = v8[1];
          if ( *(_OWORD *)&v34 == *(_OWORD *)v8 )
            break;
          ++v11;
        }
      }
      v13 = (__int64)(v8 + 2);
      v31 = (unsigned int *)a1[1];
      if ( !v8[11] )
        goto LABEL_21;
      result = sub_B99820((__int64)(v8 + 6), v31);
LABEL_13:
      ++v2;
    }
    while ( v30 != v2 );
  }
  return result;
}
