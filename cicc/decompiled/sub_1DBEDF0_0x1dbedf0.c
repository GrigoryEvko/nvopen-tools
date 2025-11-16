// Function: sub_1DBEDF0
// Address: 0x1dbedf0
//
unsigned __int64 *__fastcall sub_1DBEDF0(unsigned __int64 *a1, __int64 a2, int a3, unsigned __int64 a4)
{
  unsigned __int64 v4; // r8
  unsigned int v8; // ebx
  unsigned int v9; // r15d
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  unsigned __int64 v13; // rdx
  __int64 v14; // r13
  unsigned __int64 i; // rbx
  __int64 v16; // r9
  __int64 v17; // rcx
  unsigned int v18; // esi
  __int64 *v19; // rax
  __int64 v20; // r15
  unsigned __int64 v21; // r15
  __int64 v22; // rax
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rsi
  __int64 v29; // rcx
  unsigned int v30; // edi
  __int64 *v31; // rax
  __int64 v32; // r10
  __int64 v33; // rax
  unsigned __int64 v34; // rax
  int v36; // eax
  int v37; // eax
  __int64 v38; // r9
  __int64 v39; // rsi
  _QWORD *v40; // rcx
  _QWORD *v41; // rax
  int v42; // r11d
  int v43; // r11d
  __int128 v44; // [rsp-20h] [rbp-60h]
  __int64 v45; // [rsp-10h] [rbp-50h]
  __int64 v46; // [rsp+0h] [rbp-40h]
  unsigned __int64 v48; // [rsp+8h] [rbp-38h]
  int v49; // [rsp+8h] [rbp-38h]
  __int64 v50; // [rsp+8h] [rbp-38h]

  v4 = a4;
  v8 = a3 & 0x7FFFFFFF;
  v9 = (a3 & 0x7FFFFFFF) + 1;
  v10 = *(unsigned int *)(a2 + 408);
  if ( v9 <= (unsigned int)v10 )
    goto LABEL_2;
  v38 = v9;
  if ( v9 < v10 )
  {
    *(_DWORD *)(a2 + 408) = v9;
    goto LABEL_2;
  }
  if ( v9 <= v10 )
  {
LABEL_2:
    v11 = *(_QWORD *)(a2 + 400);
    goto LABEL_3;
  }
  if ( v9 > (unsigned __int64)*(unsigned int *)(a2 + 412) )
  {
    sub_16CD150(a2 + 400, (const void *)(a2 + 416), v9, 8, a4, v9);
    v4 = a4;
    v38 = v9;
    v10 = *(unsigned int *)(a2 + 408);
  }
  v11 = *(_QWORD *)(a2 + 400);
  v39 = *(_QWORD *)(a2 + 416);
  v40 = (_QWORD *)(v11 + 8 * v38);
  v41 = (_QWORD *)(v11 + 8 * v10);
  if ( v40 != v41 )
  {
    do
      *v41++ = v39;
    while ( v40 != v41 );
    v11 = *(_QWORD *)(a2 + 400);
  }
  *(_DWORD *)(a2 + 408) = v9;
LABEL_3:
  v48 = v4;
  *(_QWORD *)(v11 + 8LL * v8) = sub_1DBA290(a3);
  v12 = *(_QWORD *)(a2 + 272);
  v13 = v48;
  v14 = *(_QWORD *)(*(_QWORD *)(a2 + 400) + 8LL * v8);
  for ( i = v48; (*(_BYTE *)(v13 + 46) & 4) != 0; v13 = *(_QWORD *)v13 & 0xFFFFFFFFFFFFFFF8LL )
    ;
  v16 = *(_QWORD *)(v12 + 368);
  v17 = *(unsigned int *)(v12 + 384);
  if ( (_DWORD)v17 )
  {
    v18 = (v17 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
    v19 = (__int64 *)(v16 + 16LL * v18);
    v20 = *v19;
    if ( *v19 == v13 )
      goto LABEL_7;
    v36 = 1;
    while ( v20 != -8 )
    {
      v43 = v36 + 1;
      v18 = (v17 - 1) & (v36 + v18);
      v19 = (__int64 *)(v16 + 16LL * v18);
      v20 = *v19;
      if ( *v19 == v13 )
        goto LABEL_7;
      v36 = v43;
    }
  }
  v19 = (__int64 *)(v16 + 16 * v17);
LABEL_7:
  v46 = v48;
  v49 = *(_DWORD *)(v14 + 72);
  v21 = v19[1] & 0xFFFFFFFFFFFFFFF8LL;
  v22 = sub_145CBF0((__int64 *)(a2 + 296), 16, 16);
  v23 = v46;
  *(_QWORD *)(v22 + 8) = v21 | 4;
  v24 = v22;
  *(_DWORD *)v22 = v49;
  v25 = *(unsigned int *)(v14 + 72);
  if ( (unsigned int)v25 >= *(_DWORD *)(v14 + 76) )
  {
    v50 = v24;
    sub_16CD150(v14 + 64, (const void *)(v14 + 80), 0, 8, v46, v24);
    v25 = *(unsigned int *)(v14 + 72);
    v23 = v46;
    v24 = v50;
  }
  *(_QWORD *)(*(_QWORD *)(v14 + 64) + 8 * v25) = v24;
  ++*(_DWORD *)(v14 + 72);
  v26 = *(_QWORD *)(a2 + 272);
  v27 = *(_QWORD *)(*(_QWORD *)(v26 + 392) + 16LL * *(unsigned int *)(*(_QWORD *)(v23 + 24) + 48LL) + 8);
  if ( (*(_BYTE *)(v23 + 46) & 4) != 0 )
  {
    do
      i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(i + 46) & 4) != 0 );
  }
  v28 = *(_QWORD *)(v26 + 368);
  v29 = *(unsigned int *)(v26 + 384);
  if ( !(_DWORD)v29 )
  {
LABEL_19:
    v29 *= 16;
    v31 = (__int64 *)(v28 + v29);
    goto LABEL_13;
  }
  v23 = (unsigned int)(v29 - 1);
  v30 = v23 & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
  v31 = (__int64 *)(v28 + 16LL * v30);
  v32 = *v31;
  if ( i != *v31 )
  {
    v37 = 1;
    while ( v32 != -8 )
    {
      v42 = v37 + 1;
      v30 = v23 & (v37 + v30);
      v31 = (__int64 *)(v28 + 16LL * v30);
      v32 = *v31;
      if ( i == *v31 )
        goto LABEL_13;
      v37 = v42;
    }
    goto LABEL_19;
  }
LABEL_13:
  v33 = v31[1];
  a1[1] = v27;
  a1[2] = v24;
  v45 = a1[2];
  v34 = v33 & 0xFFFFFFFFFFFFFFF8LL | 4;
  *((_QWORD *)&v44 + 1) = a1[1];
  *a1 = v34;
  *(_QWORD *)&v44 = v34;
  sub_1DB8610(v14, v28, v27, v29, v23, v24, v44, v45);
  return a1;
}
