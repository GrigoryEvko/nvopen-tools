// Function: sub_729B10
// Address: 0x729b10
//
__int64 __fastcall sub_729B10(unsigned int a1, _DWORD *a2, _DWORD *a3, int a4)
{
  __int64 v4; // r8
  unsigned int v8; // ebx
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // r8
  __int64 v14; // rdx
  unsigned int v16; // eax
  __int64 v17; // r11
  __int64 v18; // rdi
  __int64 v19; // r13
  __int64 v20; // rcx
  unsigned int v21; // esi
  unsigned int v22; // eax
  unsigned int v23; // esi
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rdx
  unsigned int v29; // eax
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int128 v36; // [rsp-48h] [rbp-48h] BYREF
  __int128 v37; // [rsp-38h] [rbp-38h]

  *a3 = 0;
  v4 = 0;
  *a2 = 0;
  if ( !a1 )
    return v4;
  v4 = unk_4F07280;
  if ( !unk_4F07280 )
    return v4;
  v8 = a1;
  if ( a4 == dword_4F07B30 && a1 >= (unsigned int)qword_4F07B20 && a1 <= HIDWORD(qword_4F07B20) )
  {
    *a2 = qword_4F07B28 + a1;
    return qword_4F07B38;
  }
  v9 = a1 - 1;
  if ( !a4 && dword_4F07B00 )
  {
    v36 = 0;
    v37 = 0;
    while ( *(_DWORD *)(v4 + 28) != v9 )
    {
      v4 = *(_QWORD *)(v4 + 56);
      if ( !v4 )
      {
        DWORD2(v36) = a1;
        goto LABEL_10;
      }
    }
    *a3 = 1;
    v8 = a1 - 1;
    DWORD2(v36) = a1 - 1;
LABEL_10:
    *((_QWORD *)&v37 + 1) = v4;
    v10 = (__int64 *)bsearch(&v36, base, unk_4F07310, 8u, (__compar_fn_t)sub_727EF0);
    v11 = *v10;
    v12 = *(unsigned int *)(*v10 + 8);
    LODWORD(v10) = *(_DWORD *)(*v10 + 16);
    v13 = *(_QWORD *)(v11 + 24);
    v14 = *(_QWORD *)(v11 + 8);
    dword_4F07B30 = 0;
    qword_4F07B20 = v14;
    qword_4F07B38 = v13;
    qword_4F07B28 = (unsigned int)v10 - v12;
    *a2 = v8 + (_DWORD)v10 - v12;
    return v13;
  }
  while ( 1 )
  {
    v16 = *(_DWORD *)(v4 + 28);
    if ( v16 >= v9 )
      break;
    v4 = *(_QWORD *)(v4 + 56);
  }
  if ( v16 == v9 )
  {
    *a3 = 1;
    v8 = v16;
  }
  v17 = *(unsigned int *)(v4 + 24);
  v18 = 0;
  v19 = 0;
LABEL_16:
  if ( a4 )
  {
    if ( *(_QWORD *)(v4 + 8) )
    {
      v19 = v4;
      v18 = 0;
    }
  }
  else
  {
    v18 = 0;
  }
  v20 = *(_QWORD *)(v4 + 40);
  v21 = v17;
  if ( !v20 )
  {
LABEL_32:
    if ( a4 )
    {
      v32 = *(unsigned int *)(v19 + 32);
      v33 = *(unsigned int *)(v19 + 24);
      LODWORD(qword_4F07B20) = v21;
      v27 = v32 - v33;
    }
    else
    {
      v26 = *(unsigned int *)(v4 + 32);
      LODWORD(qword_4F07B20) = v21;
      v19 = v4;
      v27 = v26 - v17;
    }
    v28 = v27 - v18;
    v29 = *(_DWORD *)(v4 + 28);
    v4 = v19;
    goto LABEL_35;
  }
  while ( 1 )
  {
    v22 = *(_DWORD *)(v20 + 24);
    if ( v22 > v8 )
      break;
    v23 = *(_DWORD *)(v20 + 28);
    if ( !*a3 && v8 <= v23 )
    {
      v17 = v22;
      v4 = v20;
      goto LABEL_16;
    }
    v21 = v23 + 1;
    if ( *(_QWORD *)(v20 + 8) )
    {
      v18 += v21 - v22;
LABEL_22:
      v20 = *(_QWORD *)(v20 + 56);
      if ( !v20 )
        goto LABEL_32;
    }
    else
    {
      v24 = *(_QWORD *)(v20 + 40);
      if ( !v24 )
        goto LABEL_22;
      while ( *(_QWORD *)(v24 + 8) )
      {
        v25 = (unsigned int)(*(_DWORD *)(v24 + 28) + 1 - *(_DWORD *)(v24 + 24));
        v24 = *(_QWORD *)(v24 + 56);
        v18 += v25;
        if ( !v24 )
          goto LABEL_22;
      }
      v20 = *(_QWORD *)(v20 + 56);
      if ( !v20 )
        goto LABEL_32;
    }
  }
  if ( a4 )
  {
    v34 = *(unsigned int *)(v19 + 32);
    v35 = *(unsigned int *)(v19 + 24);
    LODWORD(qword_4F07B20) = v21;
    v4 = v19;
    v31 = v34 - v35;
  }
  else
  {
    v30 = *(unsigned int *)(v4 + 32);
    LODWORD(qword_4F07B20) = v21;
    v31 = v30 - v17;
  }
  v28 = v31 - v18;
  v29 = v22 - 1;
LABEL_35:
  qword_4F07B28 = v28;
  HIDWORD(qword_4F07B20) = v29;
  dword_4F07B30 = a4;
  qword_4F07B38 = v4;
  *a2 = v8 + v28;
  return v4;
}
