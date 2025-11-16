// Function: sub_25CDC80
// Address: 0x25cdc80
//
_QWORD *__fastcall sub_25CDC80(__int64 *a1, _QWORD *a2, size_t a3, __int64 a4)
{
  __int64 v5; // r12
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // rcx
  int v12; // r11d
  int v13; // r9d
  unsigned int i; // ebx
  __int64 v15; // r10
  const void *v16; // rsi
  bool v17; // al
  int v18; // eax
  unsigned int v19; // ebx
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdi
  int v24; // eax
  __int64 v25; // rsi
  int v26; // ecx
  int *v27; // r9
  int v28; // edx
  unsigned int v29; // eax
  int *v30; // rdi
  int v31; // r8d
  int v32; // ecx
  unsigned int v33; // eax
  int *v34; // rdi
  int v35; // r8d
  int v36; // edi
  int v37; // r10d
  int v38; // edi
  int v39; // r10d
  __int64 v40; // [rsp+0h] [rbp-80h]
  int v41; // [rsp+Ch] [rbp-74h]
  __int64 v42; // [rsp+10h] [rbp-70h]
  int v44; // [rsp+18h] [rbp-68h]
  __int64 v45; // [rsp+20h] [rbp-60h]
  __int64 v46; // [rsp+20h] [rbp-60h]
  int v47; // [rsp+28h] [rbp-58h]
  _QWORD *v48; // [rsp+30h] [rbp-50h]

  v5 = *a1;
  v45 = *(_QWORD *)(*a1 + 8);
  v47 = *(_DWORD *)(*a1 + 24);
  if ( !v47 )
    goto LABEL_14;
  v48 = a2;
  v8 = ((0xBF58476D1CE4E5B9LL * a4) >> 31) ^ (0xBF58476D1CE4E5B9LL * a4);
  v9 = sub_C94890(a2, a3);
  v10 = v45;
  v11 = a4;
  v12 = 1;
  v13 = v47 - 1;
  for ( i = (v47 - 1) & (((0xBF58476D1CE4E5B9LL * ((unsigned int)v8 | (v9 << 32))) >> 31) ^ (484763065 * v8));
        ;
        i = v13 & v19 )
  {
    v15 = v10 + 32LL * i;
    v16 = *(const void **)v15;
    if ( *(_QWORD *)v15 != -1 )
      break;
    if ( a2 == (_QWORD *)-1LL )
      goto LABEL_9;
LABEL_13:
    if ( *(_QWORD *)(v15 + 16) == -1 )
      goto LABEL_14;
LABEL_11:
    v19 = v12 + i;
    ++v12;
  }
  v17 = (_QWORD *)((char *)a2 + 2) == 0;
  if ( v16 == (const void *)-2LL )
    goto LABEL_8;
  if ( a3 != *(_QWORD *)(v15 + 8) )
    goto LABEL_11;
  if ( a3 )
  {
    v40 = v11;
    v41 = v12;
    v42 = v10 + 32LL * i;
    v44 = v13;
    v46 = v10;
    v18 = memcmp(a2, v16, a3);
    v10 = v46;
    v13 = v44;
    v15 = v42;
    v12 = v41;
    v11 = v40;
    v17 = v18 == 0;
LABEL_8:
    if ( v17 )
    {
LABEL_9:
      if ( v11 == *(_QWORD *)(v15 + 16) )
        goto LABEL_17;
    }
    if ( v16 != (const void *)-1LL )
      goto LABEL_11;
    goto LABEL_13;
  }
  if ( v11 != *(_QWORD *)(v15 + 16) )
    goto LABEL_11;
LABEL_17:
  if ( v15 != *(_QWORD *)(v5 + 8) + 32LL * *(unsigned int *)(v5 + 24) )
  {
    v21 = *(_QWORD *)(v5 + 32);
    v22 = v21 + 32LL * *(unsigned int *)(v15 + 24);
    if ( v22 != 32LL * *(unsigned int *)(v5 + 40) + v21 )
    {
      v23 = *((unsigned int *)a1 + 8);
      v24 = *(_DWORD *)(v22 + 24);
      v25 = a1[2];
      v26 = 2 * v24;
      v27 = (int *)(v25 + 4 * v23);
      if ( (_DWORD)v23 )
      {
        v28 = v23 - 1;
        v29 = (v23 - 1) & (74 * v24);
        v30 = (int *)(v25 + 4LL * v29);
        v31 = *v30;
        if ( v26 == *v30 )
        {
LABEL_21:
          if ( v30 != v27 )
          {
            LODWORD(v48) = 0;
            BYTE4(v48) = 1;
            return v48;
          }
        }
        else
        {
          v36 = 1;
          while ( v31 != -1 )
          {
            v37 = v36 + 1;
            v29 = v28 & (v36 + v29);
            v30 = (int *)(v25 + 4LL * v29);
            v31 = *v30;
            if ( v26 == *v30 )
              goto LABEL_21;
            v36 = v37;
          }
        }
        v32 = v26 | 1;
        v33 = v28 & (37 * v32);
        v34 = (int *)(v25 + 4LL * v33);
        v35 = *v34;
        if ( v32 == *v34 )
        {
LABEL_24:
          if ( v34 != v27 )
          {
            LODWORD(v48) = 1;
            BYTE4(v48) = 1;
            return v48;
          }
        }
        else
        {
          v38 = 1;
          while ( v35 != -1 )
          {
            v39 = v38 + 1;
            v33 = v28 & (v38 + v33);
            v34 = (int *)(v25 + 4LL * v33);
            v35 = *v34;
            if ( v32 == *v34 )
              goto LABEL_24;
            v38 = v39;
          }
        }
      }
    }
  }
LABEL_14:
  BYTE4(v48) = 0;
  return v48;
}
