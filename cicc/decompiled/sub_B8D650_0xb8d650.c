// Function: sub_B8D650
// Address: 0xb8d650
//
__int64 __fastcall sub_B8D650(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  int v4; // r13d
  __int64 v7; // r15
  __int64 v8; // rax
  int v9; // ecx
  char *v10; // r10
  __int64 v11; // r11
  __int64 v12; // rbx
  __int64 v13; // rax
  size_t v14; // r9
  unsigned int i; // ebx
  __int64 v16; // r13
  const void *v17; // r8
  bool v18; // al
  int v19; // eax
  char *v20; // rdi
  const void *v21; // rsi
  bool v22; // al
  size_t v23; // rdx
  int v24; // eax
  unsigned int v25; // ebx
  __int64 v26; // [rsp+0h] [rbp-60h]
  const void *v27; // [rsp+0h] [rbp-60h]
  int v28; // [rsp+8h] [rbp-58h]
  size_t v29; // [rsp+8h] [rbp-58h]
  size_t v30; // [rsp+10h] [rbp-50h]
  char *v31; // [rsp+10h] [rbp-50h]
  const void *v32; // [rsp+18h] [rbp-48h]
  __int64 v33; // [rsp+18h] [rbp-48h]
  char *v34; // [rsp+20h] [rbp-40h]
  int v35; // [rsp+20h] [rbp-40h]
  unsigned int v36; // [rsp+2Ch] [rbp-34h]
  int v37; // [rsp+2Ch] [rbp-34h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v7 = *(_QWORD *)(a1 + 8);
  v36 = sub_C94890(a2[2], a2[3]);
  v8 = sub_C94890(*a2, a2[1]);
  v9 = v4 - 1;
  v10 = (char *)*a2;
  v11 = 0;
  v12 = v8;
  v13 = v36;
  v37 = 1;
  v14 = a2[1];
  for ( i = (v4 - 1) & (((0xBF58476D1CE4E5B9LL * (v13 | (v12 << 32))) >> 31) ^ (484763065 * v13)); ; i = v9 & v25 )
  {
    v16 = v7 + 32LL * i;
    v17 = *(const void **)v16;
    v18 = v10 + 1 == 0;
    if ( *(_QWORD *)v16 == -1 )
      goto LABEL_9;
    v18 = v10 + 2 == 0;
    if ( v17 == (const void *)-2LL )
      goto LABEL_9;
    if ( v14 != *(_QWORD *)(v16 + 8) )
      goto LABEL_26;
    if ( v14 )
    {
      v26 = v11;
      v28 = v9;
      v30 = v14;
      v32 = *(const void **)v16;
      v34 = v10;
      v19 = memcmp(v10, *(const void **)v16, v14);
      v10 = v34;
      v17 = v32;
      v14 = v30;
      v9 = v28;
      v11 = v26;
      v18 = v19 == 0;
LABEL_9:
      if ( !v18 )
        goto LABEL_15;
    }
    v20 = (char *)a2[2];
    v21 = *(const void **)(v16 + 16);
    v22 = v20 + 1 == 0;
    if ( v21 == (const void *)-1LL || (v22 = v20 + 2 == 0, v21 == (const void *)-2LL) )
    {
      if ( v22 )
        goto LABEL_21;
    }
    else
    {
      v23 = a2[3];
      if ( v23 == *(_QWORD *)(v16 + 24) )
      {
        v27 = v17;
        v29 = v14;
        v31 = v10;
        v33 = v11;
        v35 = v9;
        if ( !v23 || (v24 = memcmp(v20, v21, v23), v9 = v35, v11 = v33, v10 = v31, v14 = v29, v17 = v27, !v24) )
        {
LABEL_21:
          *a3 = v16;
          return 1;
        }
      }
    }
LABEL_15:
    if ( v17 == (const void *)-1LL )
      break;
    if ( v17 == (const void *)-2LL && *(_QWORD *)(v16 + 16) == -2 && !v11 )
      v11 = v7 + 32LL * i;
LABEL_26:
    v25 = v37 + i;
    ++v37;
  }
  if ( *(_QWORD *)(v16 + 16) != -1 )
    goto LABEL_26;
  if ( !v11 )
    v11 = v7 + 32LL * i;
  *a3 = v11;
  return 0;
}
