// Function: sub_2DD03B0
// Address: 0x2dd03b0
//
_QWORD *__fastcall sub_2DD03B0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  const void *v5; // r13
  size_t v6; // r12
  int v7; // eax
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rsi
  int v15; // r11d
  unsigned int i; // eax
  __int64 v17; // r8
  unsigned int v18; // eax
  __int64 v19; // rbx
  __int64 *v20; // rax
  __int64 v21; // rax
  const void *v22; // r15
  size_t v23; // r14
  int v24; // eax
  unsigned int v25; // r9d
  _QWORD *v26; // r10
  __int64 v27; // rdx
  __int64 v29; // rax
  unsigned int v30; // r9d
  _QWORD *v31; // r10
  _QWORD *v32; // rcx
  __int64 *v33; // rax
  __int64 *v34; // rax
  _QWORD *v35; // [rsp+0h] [rbp-140h]
  _QWORD *v36; // [rsp+8h] [rbp-138h]
  unsigned int v37; // [rsp+14h] [rbp-12Ch]
  __int64 *v38; // [rsp+18h] [rbp-128h]
  _QWORD v39[13]; // [rsp+20h] [rbp-120h] BYREF
  __int64 v40; // [rsp+88h] [rbp-B8h]
  __int64 v41; // [rsp+90h] [rbp-B0h] BYREF
  unsigned int v42; // [rsp+98h] [rbp-A8h]
  char v43; // [rsp+110h] [rbp-30h] BYREF

  v10 = sub_BC1CD0(a4, &unk_4F82410, a3);
  v11 = *(_QWORD *)(a3 + 40);
  v12 = *(_QWORD *)(v10 + 8);
  v13 = *(unsigned int *)(v12 + 88);
  v14 = *(_QWORD *)(v12 + 72);
  if ( !(_DWORD)v13 )
    goto LABEL_26;
  v15 = 1;
  for ( i = (v13 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_501DA18 >> 9) ^ ((unsigned int)&unk_501DA18 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)))); ; i = (v13 - 1) & v18 )
  {
    v17 = v14 + 24LL * i;
    if ( *(_UNKNOWN **)v17 == &unk_501DA18 && v11 == *(_QWORD *)(v17 + 8) )
      break;
    if ( *(_QWORD *)v17 == -4096 && *(_QWORD *)(v17 + 8) == -4096 )
      goto LABEL_26;
    v18 = v15 + i;
    ++v15;
  }
  if ( v17 == v14 + 24 * v13 || (v19 = *(_QWORD *)(*(_QWORD *)(v17 + 16) + 24LL)) == 0 )
  {
LABEL_26:
    v4 = sub_B2DBE0(a3);
    v5 = *(const void **)v4;
    v6 = *(_QWORD *)(v4 + 8);
    v7 = sub_C92610();
    sub_C92740(0, v5, v6, v7);
    BUG();
  }
  v38 = (__int64 *)(v19 + 8);
  memset(v39, 0, sizeof(v39));
  v40 = 1;
  LODWORD(v39[2]) = 2;
  BYTE4(v39[3]) = 1;
  LODWORD(v39[8]) = 2;
  BYTE4(v39[9]) = 1;
  v39[1] = &v39[4];
  v39[7] = &v39[10];
  v20 = &v41;
  do
  {
    *v20 = -4096;
    v20 += 2;
  }
  while ( v20 != (__int64 *)&v43 );
  if ( (v40 & 1) == 0 )
    sub_C7D6A0(v41, 16LL * v42, 8);
  v21 = sub_B2DBE0(a3);
  v22 = *(const void **)v21;
  v23 = *(_QWORD *)(v21 + 8);
  v24 = sub_C92610();
  v25 = sub_C92740((__int64)v38, v22, v23, v24);
  v26 = (_QWORD *)(*(_QWORD *)(v19 + 8) + 8LL * v25);
  v27 = *v26;
  if ( *v26 )
  {
    if ( v27 != -8 )
      goto LABEL_15;
    --*(_DWORD *)(v19 + 24);
  }
  v36 = v26;
  v37 = v25;
  v29 = sub_C7D670(v23 + 17, 8);
  v30 = v37;
  v31 = v36;
  v32 = (_QWORD *)v29;
  if ( v23 )
  {
    v35 = (_QWORD *)v29;
    memcpy((void *)(v29 + 16), v22, v23);
    v30 = v37;
    v31 = v36;
    v32 = v35;
  }
  *((_BYTE *)v32 + v23 + 16) = 0;
  *v32 = v23;
  v32[1] = 0;
  *v31 = v32;
  ++*(_DWORD *)(v19 + 20);
  v33 = (__int64 *)(*(_QWORD *)(v19 + 8) + 8LL * (unsigned int)sub_C929D0(v38, v30));
  v27 = *v33;
  if ( *v33 == -8 || !v27 )
  {
    v34 = v33 + 1;
    do
    {
      do
        v27 = *v34++;
      while ( v27 == -8 );
    }
    while ( !v27 );
  }
LABEL_15:
  sub_2DD0370(a1, a3, *(_QWORD *)(v27 + 8));
  return a1;
}
