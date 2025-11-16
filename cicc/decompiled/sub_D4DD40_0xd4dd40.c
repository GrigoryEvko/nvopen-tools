// Function: sub_D4DD40
// Address: 0xd4dd40
//
__int64 __fastcall sub_D4DD40(__int64 **a1)
{
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 result; // rax
  unsigned int v5; // esi
  __int64 v6; // rdi
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rsi
  int v11; // eax
  _QWORD *v12; // rdx
  int v13; // edi
  unsigned int v14; // ecx
  __int64 *v15; // rax
  __int64 v16; // r8
  _QWORD *v17; // rax
  unsigned int v18; // esi
  __int64 v19; // rdi
  int v20; // r10d
  __int64 *v21; // r9
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // rcx
  int v25; // eax
  int v26; // eax
  int v27; // edx
  __int64 v28; // rcx
  unsigned __int64 v29; // rax
  __int64 v30; // r15
  int v31; // eax
  __int64 v32; // rdx
  __int64 v33; // r9
  int v34; // r14d
  int v35; // eax
  __int64 v36; // rdx
  const __m128i *v37; // rsi
  __int64 v38; // rdi
  __int64 v39; // rax
  const __m128i *v40; // rax
  const __m128i *v41; // rdi
  __m128i *v42; // rdx
  int v43; // ebx
  int v44; // eax
  int v45; // r9d
  __int64 v46; // [rsp+0h] [rbp-60h]
  __int64 *v47; // [rsp+18h] [rbp-48h] BYREF
  __int64 v48; // [rsp+20h] [rbp-40h] BYREF
  int v49; // [rsp+28h] [rbp-38h]

  v2 = (__int64)a1[1];
  v3 = *((unsigned int *)a1 + 4);
  while ( 1 )
  {
    result = v2 + 40 * v3 - 40;
    v5 = *(_DWORD *)(result + 24);
    if ( *(_DWORD *)(result + 8) == v5 )
      return result;
    while ( 1 )
    {
      v6 = *(_QWORD *)(result + 16);
      *(_DWORD *)(result + 24) = v5 + 1;
      v7 = sub_B46EC0(v6, v5);
      v8 = **a1;
      v9 = (*a1)[1];
      v10 = *(_QWORD *)(v9 + 8);
      v11 = *(_DWORD *)(v9 + 24);
      v12 = *(_QWORD **)v8;
      if ( v11 )
      {
        v13 = v11 - 1;
        v14 = (v11 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v15 = (__int64 *)(v10 + 16LL * v14);
        v16 = *v15;
        if ( v7 == *v15 )
        {
LABEL_5:
          v17 = (_QWORD *)v15[1];
          if ( v12 != v17 )
          {
            while ( v17 )
            {
              v17 = (_QWORD *)*v17;
              if ( v12 == v17 )
                goto LABEL_8;
            }
            goto LABEL_10;
          }
        }
        else
        {
          v25 = 1;
          while ( v16 != -4096 )
          {
            v45 = v25 + 1;
            v14 = v13 & (v25 + v14);
            v15 = (__int64 *)(v10 + 16LL * v14);
            v16 = *v15;
            if ( v7 == *v15 )
              goto LABEL_5;
            v25 = v45;
          }
          if ( v12 )
            goto LABEL_10;
        }
      }
      else if ( v12 )
      {
        goto LABEL_10;
      }
LABEL_8:
      v48 = v7;
      v49 = 0;
      v18 = *(_DWORD *)(v8 + 32);
      if ( !v18 )
      {
        ++*(_QWORD *)(v8 + 8);
        v47 = 0;
LABEL_50:
        v18 *= 2;
        goto LABEL_51;
      }
      v19 = *(_QWORD *)(v8 + 16);
      v20 = 1;
      v21 = 0;
      v22 = (v18 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v23 = (__int64 *)(v19 + 16LL * v22);
      v24 = *v23;
      if ( v7 != *v23 )
        break;
LABEL_10:
      result = (__int64)&a1[1][5 * *((unsigned int *)a1 + 4) - 5];
      v5 = *(_DWORD *)(result + 24);
      if ( *(_DWORD *)(result + 8) == v5 )
        return result;
    }
    while ( v24 != -4096 )
    {
      if ( v24 != -8192 || v21 )
        v23 = v21;
      v22 = (v18 - 1) & (v20 + v22);
      v24 = *(_QWORD *)(v19 + 16LL * v22);
      if ( v7 == v24 )
        goto LABEL_10;
      ++v20;
      v21 = v23;
      v23 = (__int64 *)(v19 + 16LL * v22);
    }
    if ( !v21 )
      v21 = v23;
    v26 = *(_DWORD *)(v8 + 24);
    ++*(_QWORD *)(v8 + 8);
    v27 = v26 + 1;
    v47 = v21;
    if ( 4 * (v26 + 1) >= 3 * v18 )
      goto LABEL_50;
    v28 = v7;
    if ( v18 - *(_DWORD *)(v8 + 28) - v27 > v18 >> 3 )
      goto LABEL_27;
LABEL_51:
    sub_B23080(v8 + 8, v18);
    sub_B1C700(v8 + 8, &v48, &v47);
    v28 = v48;
    v21 = v47;
    v27 = *(_DWORD *)(v8 + 24) + 1;
LABEL_27:
    *(_DWORD *)(v8 + 24) = v27;
    if ( *v21 != -4096 )
      --*(_DWORD *)(v8 + 28);
    *v21 = v28;
    *((_DWORD *)v21 + 2) = v49;
    v29 = *(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v29 == v7 + 48 )
      goto LABEL_36;
    if ( !v29 )
      BUG();
    v30 = v29 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v29 - 24) - 30 > 0xA )
    {
LABEL_36:
      v32 = *((unsigned int *)a1 + 4);
      v34 = 0;
      v33 = 0;
      v30 = 0;
      v35 = v32;
      if ( *((_DWORD *)a1 + 5) <= (unsigned int)v32 )
        goto LABEL_37;
LABEL_33:
      v2 = (__int64)a1[1];
      v36 = v2 + 40 * v32;
      if ( v36 )
      {
        *(_QWORD *)v36 = v33;
        *(_DWORD *)(v36 + 8) = v34;
        *(_QWORD *)(v36 + 16) = v30;
        *(_DWORD *)(v36 + 24) = 0;
        *(_QWORD *)(v36 + 32) = v7;
        v35 = *((_DWORD *)a1 + 4);
        v2 = (__int64)a1[1];
      }
      v3 = (unsigned int)(v35 + 1);
      *((_DWORD *)a1 + 4) = v3;
    }
    else
    {
      v31 = sub_B46E30(v30);
      v32 = *((unsigned int *)a1 + 4);
      v33 = v30;
      v34 = v31;
      v35 = v32;
      if ( *((_DWORD *)a1 + 5) > (unsigned int)v32 )
        goto LABEL_33;
LABEL_37:
      v46 = v33;
      v37 = (const __m128i *)(a1 + 3);
      v2 = sub_C8D7D0((__int64)(a1 + 1), (__int64)(a1 + 3), 0, 0x28u, (unsigned __int64 *)&v48, v33);
      v38 = 40LL * *((unsigned int *)a1 + 4);
      v39 = v38 + v2;
      if ( v38 + v2 )
      {
        *(_DWORD *)(v39 + 8) = v34;
        *(_QWORD *)(v39 + 16) = v30;
        *(_QWORD *)v39 = v46;
        *(_DWORD *)(v39 + 24) = 0;
        *(_QWORD *)(v39 + 32) = v7;
        v38 = 40LL * *((unsigned int *)a1 + 4);
      }
      v40 = (const __m128i *)a1[1];
      v41 = (const __m128i *)((char *)v40 + v38);
      if ( v40 != v41 )
      {
        v42 = (__m128i *)v2;
        do
        {
          if ( v42 )
          {
            *v42 = _mm_loadu_si128(v40);
            v42[1] = _mm_loadu_si128(v40 + 1);
            v42[2].m128i_i64[0] = v40[2].m128i_i64[0];
          }
          v40 = (const __m128i *)((char *)v40 + 40);
          v42 = (__m128i *)((char *)v42 + 40);
        }
        while ( v41 != v40 );
        v41 = (const __m128i *)a1[1];
      }
      v43 = v48;
      if ( v41 != v37 )
        _libc_free(v41, v37);
      v44 = *((_DWORD *)a1 + 4);
      a1[1] = (__int64 *)v2;
      *((_DWORD *)a1 + 5) = v43;
      v3 = (unsigned int)(v44 + 1);
      *((_DWORD *)a1 + 4) = v3;
    }
  }
}
