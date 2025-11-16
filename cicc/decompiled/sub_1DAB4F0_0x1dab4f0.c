// Function: sub_1DAB4F0
// Address: 0x1dab4f0
//
char __fastcall sub_1DAB4F0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  unsigned int *v10; // rax
  __int64 v11; // rax
  int v12; // edi
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r14
  __int64 v17; // rcx
  __int64 v18; // rsi
  __int64 v19; // rcx
  _QWORD *v20; // rax
  __int64 v21; // r8
  __int64 v22; // rsi
  __int64 v23; // rdi
  __int64 v24; // rcx
  __int64 v25; // r14
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rdx
  __int64 v30; // rsi
  int v31; // eax
  __int64 v32; // rdx
  __int64 v33; // r12
  __int64 v34; // rdx
  __int64 v35; // rax
  unsigned __int64 v36; // rax
  __int64 v37; // rax

  v7 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16;
  v8 = *(_QWORD *)v7;
  v9 = *(unsigned int *)(v7 + 12);
  if ( *(_DWORD *)(*(_QWORD *)a1 + 80LL) )
    v10 = (unsigned int *)(v8 + 4 * v9 + 144);
  else
    v10 = (unsigned int *)(v8 + 4 * v9 + 64);
  *v10 = a2;
  v11 = *(unsigned int *)(a1 + 16);
  if ( *(_DWORD *)(*(_QWORD *)a1 + 80LL) )
  {
    v12 = *(_DWORD *)(a1 + 16);
    v13 = *(_QWORD *)(a1 + 8) + 16 * v11 - 16;
    v14 = *(_QWORD *)v13;
    v15 = 16LL * *(unsigned int *)(v13 + 12);
    v16 = *(_QWORD *)(*(_QWORD *)v13 + v15 + 8);
    v17 = (unsigned int)(*(_DWORD *)(v13 + 12) + 1);
    if ( *(_DWORD *)(v13 + 8) <= (unsigned int)v17 )
    {
      v18 = (unsigned int)(v12 - 1);
      v35 = sub_3945FF0(a1 + 8, v18);
      if ( !v35 )
        goto LABEL_16;
      v36 = v35 & 0xFFFFFFFFFFFFFFC0LL;
      if ( ((a2 ^ *(_DWORD *)(v36 + 144)) & 0x7FFFFFFF) != 0
        || ((*(_BYTE *)(v36 + 147) ^ HIBYTE(a2)) & 0x80u) != 0
        || *(_QWORD *)v36 != v16 )
      {
        goto LABEL_16;
      }
      v37 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16;
      v14 = *(_QWORD *)v37;
      v15 = 16LL * *(unsigned int *)(v37 + 12);
    }
    else
    {
      if ( ((a2 ^ *(_DWORD *)(v14 + 4 * v17 + 144)) & 0x7FFFFFFF) != 0 )
        goto LABEL_8;
      v18 = HIBYTE(a2);
      LOBYTE(v18) = *(_BYTE *)(v14 + 4 * (v17 + 36) + 3) ^ HIBYTE(a2);
      if ( (v18 & 0x80u) != 0LL )
        goto LABEL_8;
      v19 = 16 * v17;
      if ( *(_QWORD *)(v14 + v19) != v16 )
        goto LABEL_8;
    }
LABEL_15:
    v25 = *(_QWORD *)(v14 + v15);
    sub_1DAB460(a1, v18, v14, v19, a5);
    v26 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16;
    *(_QWORD *)(*(_QWORD *)v26 + 16LL * *(unsigned int *)(v26 + 12)) = v25;
LABEL_16:
    v27 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16;
    v14 = *(_QWORD *)v27;
    v15 = 16LL * *(unsigned int *)(v27 + 12);
    goto LABEL_8;
  }
  v22 = *(_QWORD *)(a1 + 8) + 16 * v11 - 16;
  v14 = *(_QWORD *)v22;
  v15 = 16LL * *(unsigned int *)(v22 + 12);
  v23 = *(_QWORD *)(*(_QWORD *)v22 + v15 + 8);
  v24 = (unsigned int)(*(_DWORD *)(v22 + 12) + 1);
  if ( *(_DWORD *)(v22 + 8) > (unsigned int)v24 )
  {
    a5 = v24 + 16;
    if ( ((a2 ^ *(_DWORD *)(v14 + 4 * v24 + 64)) & 0x7FFFFFFF) == 0 )
    {
      v18 = HIBYTE(a2);
      LOBYTE(v18) = *(_BYTE *)(v14 + 4 * a5 + 3) ^ HIBYTE(a2);
      if ( (v18 & 0x80u) == 0LL )
      {
        v19 = 16 * v24;
        if ( v23 == *(_QWORD *)(v14 + v19) )
          goto LABEL_15;
      }
    }
  }
LABEL_8:
  LOBYTE(v20) = sub_1DA98F0(a1, *(_QWORD *)(v14 + v15), a2);
  if ( !(_BYTE)v20 )
    return (char)v20;
  v28 = *(_QWORD *)(a1 + 8);
  v29 = *(unsigned int *)(a1 + 16);
  v30 = v28 + 16 * v29 - 16;
  v31 = *(_DWORD *)(v30 + 12);
  if ( !v31 )
  {
    LODWORD(v21) = *(_DWORD *)(*(_QWORD *)a1 + 80LL);
LABEL_30:
    v30 = (unsigned int)v21;
    sub_3945E40(a1 + 8, (unsigned int)v21);
    goto LABEL_21;
  }
  if ( !(_DWORD)v29 || *(_DWORD *)(v28 + 12) >= *(_DWORD *)(v28 + 8) )
  {
    v21 = *(unsigned int *)(*(_QWORD *)a1 + 80LL);
    if ( (_DWORD)v21 )
      goto LABEL_30;
  }
  *(_DWORD *)(v30 + 12) = v31 - 1;
LABEL_21:
  v32 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16;
  v33 = *(_QWORD *)(*(_QWORD *)v32 + 16LL * *(unsigned int *)(v32 + 12));
  sub_1DAB460(a1, v30, v32, v28, v21);
  v34 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16;
  v20 = (_QWORD *)(*(_QWORD *)v34 + 16LL * *(unsigned int *)(v34 + 12));
  *v20 = v33;
  return (char)v20;
}
