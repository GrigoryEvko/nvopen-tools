// Function: sub_1EE69C0
// Address: 0x1ee69c0
//
_QWORD *__fastcall sub_1EE69C0(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 i; // rdx
  __int64 v6; // rdi
  __int64 v7; // rdx
  unsigned int v8; // ecx
  __int64 *v9; // rax
  __int64 v10; // r9
  __int64 v11; // r9
  _QWORD *result; // rax
  _QWORD *v13; // r8
  unsigned __int64 v14; // r9
  __int64 v15; // r13
  __int64 v16; // r15
  __int64 *v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // rsi
  unsigned int v20; // edi
  unsigned int v21; // ecx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // r12
  unsigned __int64 v26; // rdx
  unsigned int v27; // eax
  __int64 v28; // r10
  __int64 v29; // r11
  unsigned int v30; // eax
  __int64 v31; // rcx
  int v32; // eax
  __int64 v33; // rax
  __int64 v34; // rcx
  int v35; // eax
  char *v36; // rcx
  _QWORD *v37; // rax
  __int64 v38; // r15
  __int64 v39; // rdi
  _QWORD *v40; // rsi
  _QWORD *v41; // rdx
  int v42; // r10d
  unsigned int v43; // [rsp+14h] [rbp-4Ch]
  __int64 v44; // [rsp+20h] [rbp-40h]
  _QWORD *v45; // [rsp+20h] [rbp-40h]
  _QWORD *dest; // [rsp+28h] [rbp-38h]
  _QWORD *desta; // [rsp+28h] [rbp-38h]

  for ( i = *(_QWORD *)(a3 + 272); (*(_BYTE *)(a2 + 46) & 4) != 0; a2 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL )
    ;
  v6 = *(_QWORD *)(i + 368);
  v7 = *(unsigned int *)(i + 384);
  if ( (_DWORD)v7 )
  {
    v8 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v6 + 16LL * v8);
    v10 = *v9;
    if ( *v9 == a2 )
      goto LABEL_5;
    v32 = 1;
    while ( v10 != -8 )
    {
      v42 = v32 + 1;
      v8 = (v7 - 1) & (v32 + v8);
      v9 = (__int64 *)(v6 + 16LL * v8);
      v10 = *v9;
      if ( *v9 == a2 )
        goto LABEL_5;
      v32 = v42;
    }
  }
  v9 = (__int64 *)(v6 + 16 * v7);
LABEL_5:
  v11 = v9[1];
  result = (_QWORD *)*(unsigned int *)(a1 + 88);
  v13 = *(_QWORD **)(a1 + 80);
  if ( (_DWORD)result )
  {
    v14 = v11 & 0xFFFFFFFFFFFFFFF8LL;
    v15 = v14;
    do
    {
      v25 = *(unsigned int *)v13;
      if ( (int)v25 >= 0 )
      {
        v16 = *(_QWORD *)(*(_QWORD *)(a3 + 672) + 8 * v25);
        goto LABEL_8;
      }
      v26 = *(unsigned int *)(a3 + 408);
      v27 = v25 & 0x7FFFFFFF;
      v28 = v25 & 0x7FFFFFFF;
      v29 = 8 * v28;
      if ( ((unsigned int)v25 & 0x7FFFFFFF) < (unsigned int)v26 )
      {
        v16 = *(_QWORD *)(*(_QWORD *)(a3 + 400) + 8LL * v27);
        if ( v16 )
          goto LABEL_9;
      }
      v30 = v27 + 1;
      if ( (unsigned int)v26 < v30 )
      {
        v38 = v30;
        if ( v30 >= v26 )
        {
          if ( v30 > v26 )
          {
            if ( v30 > (unsigned __int64)*(unsigned int *)(a3 + 412) )
            {
              v43 = v30;
              v45 = v13;
              sub_16CD150(a3 + 400, (const void *)(a3 + 416), v30, 8, (int)v13, v14);
              v26 = *(unsigned int *)(a3 + 408);
              v28 = v25 & 0x7FFFFFFF;
              v30 = v43;
              v13 = v45;
              v29 = 8 * v28;
            }
            v31 = *(_QWORD *)(a3 + 400);
            v39 = *(_QWORD *)(a3 + 416);
            v40 = (_QWORD *)(v31 + 8 * v38);
            v41 = (_QWORD *)(v31 + 8 * v26);
            if ( v40 != v41 )
            {
              do
                *v41++ = v39;
              while ( v40 != v41 );
              v31 = *(_QWORD *)(a3 + 400);
            }
            *(_DWORD *)(a3 + 408) = v30;
            goto LABEL_22;
          }
        }
        else
        {
          *(_DWORD *)(a3 + 408) = v30;
        }
      }
      v31 = *(_QWORD *)(a3 + 400);
LABEL_22:
      desta = v13;
      v44 = v28;
      *(_QWORD *)(v31 + v29) = sub_1DBA290(v25);
      v16 = *(_QWORD *)(*(_QWORD *)(a3 + 400) + 8 * v44);
      sub_1DBB110((_QWORD *)a3, v16);
      v13 = desta;
LABEL_8:
      if ( !v16 )
        goto LABEL_15;
LABEL_9:
      dest = v13;
      v17 = (__int64 *)sub_1DB3C70((__int64 *)v16, v15);
      v13 = dest;
      v18 = v17;
      v19 = *(_QWORD *)v16 + 24LL * *(unsigned int *)(v16 + 8);
      if ( v17 == (__int64 *)v19 )
        goto LABEL_15;
      v20 = *(_DWORD *)(v15 + 24);
      v21 = *(_DWORD *)((*v17 & 0xFFFFFFFFFFFFFFF8LL) + 24);
      if ( (unsigned __int64)(v21 | (*v17 >> 1) & 3) > v20 )
      {
        LOBYTE(v22) = 0;
      }
      else
      {
        v22 = v17[1];
        if ( v15 == (v18[1] & 0xFFFFFFFFFFFFFFF8LL) )
        {
          if ( (__int64 *)v19 == v18 + 3 )
            goto LABEL_14;
          v21 = *(_DWORD *)((v18[3] & 0xFFFFFFFFFFFFFFF8LL) + 24);
          v18 += 3;
        }
      }
      if ( v20 >= v21 )
        v22 = v18[1];
LABEL_14:
      if ( (((unsigned __int8)v22 ^ 6) & 6) == 0 )
      {
        v33 = *(unsigned int *)(a1 + 168);
        if ( (unsigned int)v33 >= *(_DWORD *)(a1 + 172) )
        {
          sub_16CD150(a1 + 160, (const void *)(a1 + 176), 0, 8, (int)dest, v14);
          v33 = *(unsigned int *)(a1 + 168);
          v13 = dest;
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 160) + 8 * v33) = *v13;
        v34 = *(unsigned int *)(a1 + 88);
        v23 = *(_QWORD *)(a1 + 80);
        ++*(_DWORD *)(a1 + 168);
        v35 = v34;
        v36 = (char *)(v23 + 8 * v34);
        if ( v36 != (char *)(v13 + 1) )
        {
          v37 = memmove(v13, v13 + 1, v36 - (char *)(v13 + 1));
          v23 = *(_QWORD *)(a1 + 80);
          v13 = v37;
          v35 = *(_DWORD *)(a1 + 88);
        }
        v24 = (unsigned int)(v35 - 1);
        *(_DWORD *)(a1 + 88) = v24;
        goto LABEL_16;
      }
LABEL_15:
      v23 = *(_QWORD *)(a1 + 80);
      v24 = *(unsigned int *)(a1 + 88);
      ++v13;
LABEL_16:
      result = (_QWORD *)(v23 + 8 * v24);
    }
    while ( v13 != result );
  }
  return result;
}
