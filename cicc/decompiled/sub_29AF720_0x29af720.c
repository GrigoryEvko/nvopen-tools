// Function: sub_29AF720
// Address: 0x29af720
//
__int64 __fastcall sub_29AF720(__int64 a1, __int64 a2)
{
  __int64 v4; // r12
  char v5; // cl
  __int64 v6; // rdi
  int v7; // esi
  unsigned int v8; // edx
  _QWORD *v9; // rax
  __int64 v10; // r9
  __int64 *v11; // r12
  __int64 result; // rax
  unsigned int v13; // esi
  unsigned int v14; // eax
  _QWORD *v15; // r8
  int v16; // edx
  unsigned int v17; // edi
  __int64 v18; // r14
  unsigned __int8 v19; // al
  unsigned __int8 **v20; // rdi
  __int64 v21; // rax
  __int64 v22; // r13
  int v23; // ecx
  __int64 v24; // r15
  unsigned __int8 v25; // al
  __int64 *v26; // r14
  int v27; // r9d
  __int64 v28; // rsi
  __int64 v29; // rbx
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r10
  int v34; // r10d
  __int64 v35; // rcx
  int v36; // edx
  unsigned int v37; // eax
  __int64 v38; // rsi
  __int64 v39; // rcx
  int v40; // edx
  unsigned int v41; // eax
  __int64 v42; // rsi
  int v43; // r9d
  _QWORD *v44; // rdi
  int v45; // edx
  int v46; // edx
  int v47; // r9d
  int v48; // [rsp+8h] [rbp-38h]
  int v49; // [rsp+Ch] [rbp-34h]

  v4 = *(_QWORD *)a1;
  v5 = *(_BYTE *)(*(_QWORD *)a1 + 8LL) & 1;
  if ( v5 )
  {
    v6 = v4 + 16;
    v7 = 3;
  }
  else
  {
    v13 = *(_DWORD *)(v4 + 24);
    v6 = *(_QWORD *)(v4 + 16);
    if ( !v13 )
    {
      v14 = *(_DWORD *)(v4 + 8);
      ++*(_QWORD *)v4;
      v15 = 0;
      v16 = (v14 >> 1) + 1;
LABEL_9:
      v17 = 3 * v13;
      goto LABEL_10;
    }
    v7 = v13 - 1;
  }
  v8 = v7 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (_QWORD *)(v6 + 16LL * v8);
  v10 = *v9;
  if ( a2 == *v9 )
  {
LABEL_4:
    v11 = v9 + 1;
    result = v9[1];
    if ( result )
      return result;
    goto LABEL_15;
  }
  v34 = 1;
  v15 = 0;
  while ( v10 != -4096 )
  {
    if ( !v15 && v10 == -8192 )
      v15 = v9;
    v8 = v7 & (v34 + v8);
    v9 = (_QWORD *)(v6 + 16LL * v8);
    v10 = *v9;
    if ( a2 == *v9 )
      goto LABEL_4;
    ++v34;
  }
  v17 = 12;
  v13 = 4;
  if ( !v15 )
    v15 = v9;
  v14 = *(_DWORD *)(v4 + 8);
  ++*(_QWORD *)v4;
  v16 = (v14 >> 1) + 1;
  if ( !v5 )
  {
    v13 = *(_DWORD *)(v4 + 24);
    goto LABEL_9;
  }
LABEL_10:
  if ( 4 * v16 >= v17 )
  {
    sub_29AF300(v4, 2 * v13);
    if ( (*(_BYTE *)(v4 + 8) & 1) != 0 )
    {
      v35 = v4 + 16;
      v36 = 3;
    }
    else
    {
      v45 = *(_DWORD *)(v4 + 24);
      v35 = *(_QWORD *)(v4 + 16);
      if ( !v45 )
        goto LABEL_63;
      v36 = v45 - 1;
    }
    v37 = v36 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v15 = (_QWORD *)(v35 + 16LL * v37);
    v38 = *v15;
    if ( a2 != *v15 )
    {
      v47 = 1;
      v44 = 0;
      while ( v38 != -4096 )
      {
        if ( v38 == -8192 && !v44 )
          v44 = v15;
        v37 = v36 & (v47 + v37);
        v15 = (_QWORD *)(v35 + 16LL * v37);
        v38 = *v15;
        if ( a2 == *v15 )
          goto LABEL_33;
        ++v47;
      }
      goto LABEL_39;
    }
LABEL_33:
    v14 = *(_DWORD *)(v4 + 8);
    goto LABEL_12;
  }
  if ( v13 - *(_DWORD *)(v4 + 12) - v16 <= v13 >> 3 )
  {
    sub_29AF300(v4, v13);
    if ( (*(_BYTE *)(v4 + 8) & 1) != 0 )
    {
      v39 = v4 + 16;
      v40 = 3;
      goto LABEL_36;
    }
    v46 = *(_DWORD *)(v4 + 24);
    v39 = *(_QWORD *)(v4 + 16);
    if ( v46 )
    {
      v40 = v46 - 1;
LABEL_36:
      v41 = v40 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = (_QWORD *)(v39 + 16LL * v41);
      v42 = *v15;
      if ( a2 != *v15 )
      {
        v43 = 1;
        v44 = 0;
        while ( v42 != -4096 )
        {
          if ( !v44 && v42 == -8192 )
            v44 = v15;
          v41 = v40 & (v43 + v41);
          v15 = (_QWORD *)(v39 + 16LL * v41);
          v42 = *v15;
          if ( a2 == *v15 )
            goto LABEL_33;
          ++v43;
        }
LABEL_39:
        if ( v44 )
          v15 = v44;
        goto LABEL_33;
      }
      goto LABEL_33;
    }
LABEL_63:
    *(_DWORD *)(v4 + 8) = (2 * (*(_DWORD *)(v4 + 8) >> 1) + 2) | *(_DWORD *)(v4 + 8) & 1;
    BUG();
  }
LABEL_12:
  *(_DWORD *)(v4 + 8) = (2 * (v14 >> 1) + 2) | v14 & 1;
  if ( *v15 != -4096 )
    --*(_DWORD *)(v4 + 12);
  *v15 = a2;
  v11 = v15 + 1;
  v15[1] = 0;
LABEL_15:
  v18 = a2 - 16;
  v19 = *(_BYTE *)(a2 - 16);
  if ( (v19 & 2) != 0 )
    v20 = *(unsigned __int8 ***)(a2 - 32);
  else
    v20 = (unsigned __int8 **)(v18 - 8LL * ((v19 >> 2) & 0xF));
  v21 = sub_B00540(*v20, **(_QWORD **)(a1 + 8), *(_QWORD *)(a1 + 16), *(_QWORD *)(a1 + 24));
  v22 = *(_QWORD *)(a1 + 32);
  v23 = *(_DWORD *)(a2 + 4);
  v24 = v21;
  v25 = *(_BYTE *)(a2 - 16);
  if ( (v25 & 2) != 0 )
  {
    v26 = *(__int64 **)(a2 - 32);
    v27 = *(_DWORD *)(a2 + 16);
  }
  else
  {
    v27 = *(_DWORD *)(a2 + 16);
    v26 = (__int64 *)(v18 - 8LL * ((v25 >> 2) & 0xF));
  }
  v28 = v26[3];
  v29 = v26[2];
  v30 = v26[1];
  if ( v30 )
  {
    v48 = v27;
    v49 = v23;
    v31 = sub_B91420(v26[1]);
    v23 = v49;
    v27 = v48;
    v33 = v32;
    v30 = v31;
  }
  else
  {
    v33 = 0;
  }
  result = sub_ADFB30(v22, v24, v30, v33, v29, v27, v28, 0, 0, v23);
  *v11 = result;
  return result;
}
