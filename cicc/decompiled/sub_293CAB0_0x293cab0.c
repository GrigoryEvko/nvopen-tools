// Function: sub_293CAB0
// Address: 0x293cab0
//
unsigned __int64 *__fastcall sub_293CAB0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v5; // r13
  __int64 v7; // rbx
  unsigned __int8 *v8; // r8
  __int64 v9; // r9
  unsigned __int64 v10; // rdx
  unsigned __int64 *v11; // rax
  unsigned __int64 *v12; // r15
  __int64 v13; // rax
  __int64 v15; // rbx
  __int64 v16; // rdx
  int v17; // eax
  _QWORD *v18; // rdi
  unsigned __int8 *v19; // r12
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rax
  int v22; // r12d
  __int64 v23; // rcx
  __int64 v24; // rdx
  const void *v25; // rsi
  __int64 v26; // rax
  unsigned __int64 *result; // rax
  unsigned __int64 *v28; // rax
  unsigned __int64 *v29; // rdi
  unsigned __int64 v30; // rdi
  int v31; // r12d
  unsigned __int64 *v32; // rax
  unsigned __int64 *v33; // [rsp+0h] [rbp-80h]
  unsigned __int64 *v34; // [rsp+0h] [rbp-80h]
  __int64 v35; // [rsp+18h] [rbp-68h]
  __int64 v36; // [rsp+20h] [rbp-60h]
  __int64 v37; // [rsp+28h] [rbp-58h]
  __int64 v38; // [rsp+28h] [rbp-58h]
  const __m128i *v39; // [rsp+38h] [rbp-48h] BYREF
  unsigned __int64 v40[8]; // [rsp+40h] [rbp-40h] BYREF

  v5 = a2;
  v7 = a1;
  sub_293A860(a2, a3);
  v10 = *(_QWORD *)(a4 + 16);
  v11 = *(unsigned __int64 **)(a1 + 16);
  v40[0] = a2;
  v12 = (unsigned __int64 *)(a1 + 8);
  v40[1] = v10;
  if ( !v11 )
    goto LABEL_10;
  do
  {
    while ( a2 <= v11[4] && (a2 != v11[4] || v10 <= v11[5]) )
    {
      v12 = v11;
      v11 = (unsigned __int64 *)v11[2];
      if ( !v11 )
        goto LABEL_8;
    }
    v11 = (unsigned __int64 *)v11[3];
  }
  while ( v11 );
LABEL_8:
  if ( (unsigned __int64 *)(a1 + 8) == v12 || a2 < v12[4] || a2 == v12[4] && v10 < v12[5] )
  {
LABEL_10:
    v39 = (const __m128i *)v40;
    v12 = sub_293C9D0((_QWORD *)a1, v12, &v39);
  }
  v36 = (__int64)(v12 + 6);
  v13 = *((unsigned int *)v12 + 14);
  if ( (_DWORD)v13 )
  {
    v37 = 8 * v13;
    v15 = 0;
    v35 = a1 + 328;
    do
    {
      v19 = *(unsigned __int8 **)(v12[6] + v15);
      if ( v19 )
      {
        v8 = *(unsigned __int8 **)(*(_QWORD *)a3 + v15);
        if ( v8 != v19 )
        {
          if ( *v8 > 0x1Cu )
          {
            sub_BD6B90(*(unsigned __int8 **)(*(_QWORD *)a3 + v15), v19);
            v8 = *(unsigned __int8 **)(*(_QWORD *)a3 + v15);
          }
          sub_BD84D0((__int64)v19, (__int64)v8);
          v16 = *(unsigned int *)(a1 + 336);
          v17 = v16;
          if ( *(_DWORD *)(a1 + 340) <= (unsigned int)v16 )
          {
            v28 = (unsigned __int64 *)sub_C8D7D0(v35, a1 + 344, 0, 0x18u, v40, v9);
            v29 = &v28[3 * *(unsigned int *)(a1 + 336)];
            if ( v29 )
            {
              *v29 = 6;
              v29[1] = 0;
              v29[2] = (unsigned __int64)v19;
              if ( v19 != (unsigned __int8 *)-8192LL && v19 != (unsigned __int8 *)-4096LL )
              {
                v33 = v28;
                sub_BD73F0((__int64)v29);
                v28 = v33;
              }
            }
            v34 = v28;
            sub_F17F80(v35, v28);
            v30 = *(_QWORD *)(a1 + 328);
            v31 = v40[0];
            v32 = v34;
            if ( a1 + 344 != v30 )
            {
              _libc_free(v30);
              v32 = v34;
            }
            ++*(_DWORD *)(a1 + 336);
            *(_QWORD *)(a1 + 328) = v32;
            *(_DWORD *)(a1 + 340) = v31;
          }
          else
          {
            v18 = (_QWORD *)(*(_QWORD *)(a1 + 328) + 24 * v16);
            if ( v18 )
            {
              *v18 = 6;
              v18[1] = 0;
              v18[2] = v19;
              if ( v19 != (unsigned __int8 *)-8192LL && v19 != (unsigned __int8 *)-4096LL )
                sub_BD73F0((__int64)v18);
              v17 = *(_DWORD *)(a1 + 336);
            }
            *(_DWORD *)(a1 + 336) = v17 + 1;
          }
        }
      }
      v15 += 8;
    }
    while ( v37 != v15 );
    v7 = a1;
    v5 = a2;
  }
  if ( a3 != v36 )
  {
    v20 = *(unsigned int *)(a3 + 8);
    v21 = *((unsigned int *)v12 + 14);
    v22 = *(_DWORD *)(a3 + 8);
    if ( v20 <= v21 )
    {
      if ( *(_DWORD *)(a3 + 8) )
        memmove((void *)v12[6], *(const void **)a3, 8 * v20);
    }
    else
    {
      if ( v20 > *((unsigned int *)v12 + 15) )
      {
        *((_DWORD *)v12 + 14) = 0;
        sub_C8D5F0(v36, v12 + 8, v20, 8u, (__int64)v8, v9);
        v20 = *(unsigned int *)(a3 + 8);
        v23 = 0;
      }
      else
      {
        v23 = 8 * v21;
        if ( *((_DWORD *)v12 + 14) )
        {
          v38 = 8 * v21;
          memmove((void *)v12[6], *(const void **)a3, 8 * v21);
          v20 = *(unsigned int *)(a3 + 8);
          v23 = v38;
        }
      }
      v24 = 8 * v20;
      v25 = (const void *)(*(_QWORD *)a3 + v23);
      if ( v25 != (const void *)(v24 + *(_QWORD *)a3) )
        memcpy((void *)(v23 + v12[6]), v25, v24 - v23);
    }
    *((_DWORD *)v12 + 14) = v22;
  }
  v26 = *(unsigned int *)(v7 + 56);
  if ( v26 + 1 > (unsigned __int64)*(unsigned int *)(v7 + 60) )
  {
    sub_C8D5F0(v7 + 48, (const void *)(v7 + 64), v26 + 1, 0x10u, (__int64)v8, v9);
    v26 = *(unsigned int *)(v7 + 56);
  }
  result = (unsigned __int64 *)(*(_QWORD *)(v7 + 48) + 16 * v26);
  *result = v5;
  result[1] = v36;
  ++*(_DWORD *)(v7 + 56);
  return result;
}
