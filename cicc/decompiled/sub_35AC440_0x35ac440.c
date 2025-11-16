// Function: sub_35AC440
// Address: 0x35ac440
//
char __fastcall sub_35AC440(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  __int64 v8; // rbx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rcx
  __int64 v11; // r13
  unsigned __int64 v12; // rdx
  __int64 v13; // rbx
  unsigned __int64 v14; // rdi
  int v15; // ecx
  __int64 v16; // rax
  int v17; // edx
  __int64 v18; // rbx
  __int64 v19; // rbx
  __int64 *v20; // rdi
  __int64 v21; // rdx
  int v22; // ecx
  __int64 v23; // rax

  v7 = a1 + 216;
  v8 = *(_QWORD *)(a2 + 48);
  v9 = *(unsigned int *)(a1 + 224);
  v10 = *(unsigned int *)(a1 + 228);
  v11 = *(_QWORD *)(v8 + 672);
  v12 = v9 + 1;
  if ( v11 )
  {
    if ( v12 > v10 )
    {
      sub_C8D5F0(a1 + 216, (const void *)(a1 + 232), v12, 8u, a5, a6);
      v9 = *(unsigned int *)(a1 + 224);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 216) + 8 * v9) = v11;
    ++*(_DWORD *)(a1 + 224);
    v13 = *(_QWORD *)(v8 + 680);
    if ( *(_DWORD *)(v13 + 120) )
    {
LABEL_5:
      v9 = *(unsigned int *)(a1 + 272);
      if ( v9 + 1 > *(unsigned int *)(a1 + 276) )
      {
        sub_C8D5F0(a1 + 264, (const void *)(a1 + 280), v9 + 1, 8u, a5, a6);
        v9 = *(unsigned int *)(a1 + 272);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 264) + 8 * v9) = v13;
      ++*(_DWORD *)(a1 + 272);
      return v9;
    }
    v9 = *(_QWORD *)(v13 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    v14 = v9;
    if ( v9 == v13 + 48 )
      return v9;
    if ( !v9 )
      BUG();
    v15 = *(_DWORD *)(v9 + 44);
    v16 = *(_QWORD *)v9;
    v17 = *(_DWORD *)(v14 + 44) & 0xFFFFFF;
    if ( (v16 & 4) != 0 )
    {
      if ( (v15 & 4) != 0 )
        goto LABEL_14;
    }
    else if ( (v15 & 4) != 0 )
    {
      while ( 1 )
      {
        v14 = v16 & 0xFFFFFFFFFFFFFFF8LL;
        LOBYTE(v17) = *(_DWORD *)((v16 & 0xFFFFFFFFFFFFFFF8LL) + 44);
        if ( (v17 & 4) == 0 )
          break;
        v16 = *(_QWORD *)v14;
      }
    }
    if ( (v17 & 8) != 0 )
    {
      LOBYTE(v9) = sub_2E88A90(v14, 32, 1);
LABEL_15:
      if ( !(_BYTE)v9 )
        return v9;
      goto LABEL_5;
    }
LABEL_14:
    v9 = (*(_QWORD *)(*(_QWORD *)(v14 + 16) + 24LL) >> 5) & 1LL;
    goto LABEL_15;
  }
  v18 = *(_QWORD *)(a2 + 328);
  if ( v12 > v10 )
  {
    sub_C8D5F0(a1 + 216, (const void *)(a1 + 232), v12, 8u, a5, a6);
    v9 = *(unsigned int *)(a1 + 224);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 216) + 8 * v9) = v18;
  ++*(_DWORD *)(a1 + 224);
  v19 = *(_QWORD *)(a2 + 328);
  if ( v19 != a2 + 320 )
  {
    if ( *(_BYTE *)(v19 + 235) )
      goto LABEL_36;
    while ( 1 )
    {
      v9 = *(_QWORD *)(v19 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      v20 = (__int64 *)v9;
      if ( v9 == v19 + 48 )
        goto LABEL_34;
      if ( !v9 )
        BUG();
      v21 = *(_QWORD *)v9;
      v22 = *(_DWORD *)(v9 + 44);
      if ( (*(_QWORD *)v9 & 4) != 0 )
      {
        if ( (v22 & 4) != 0 )
          goto LABEL_40;
      }
      else if ( (v22 & 4) != 0 )
      {
        while ( 1 )
        {
          v20 = (__int64 *)(v21 & 0xFFFFFFFFFFFFFFF8LL);
          v22 = *(_DWORD *)((v21 & 0xFFFFFFFFFFFFFFF8LL) + 44) & 0xFFFFFF;
          if ( (*(_DWORD *)((v21 & 0xFFFFFFFFFFFFFFF8LL) + 44) & 4) == 0 )
            break;
          v21 = *v20;
        }
      }
      if ( (v22 & 8) != 0 )
      {
        LOBYTE(v9) = sub_2E88A90((__int64)v20, 32, 1);
        goto LABEL_30;
      }
LABEL_40:
      v9 = (*(_QWORD *)(v20[2] + 24) >> 5) & 1LL;
LABEL_30:
      if ( (_BYTE)v9 )
      {
        v9 = *(unsigned int *)(a1 + 272);
        if ( v9 + 1 > *(unsigned int *)(a1 + 276) )
        {
          sub_C8D5F0(a1 + 264, (const void *)(a1 + 280), v9 + 1, 8u, a5, a6);
          v9 = *(unsigned int *)(a1 + 272);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 264) + 8 * v9) = v19;
        ++*(_DWORD *)(a1 + 272);
      }
LABEL_34:
      v19 = *(_QWORD *)(v19 + 8);
      if ( a2 + 320 == v19 )
        return v9;
      if ( *(_BYTE *)(v19 + 235) )
      {
LABEL_36:
        v23 = *(unsigned int *)(a1 + 224);
        if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 228) )
        {
          sub_C8D5F0(v7, (const void *)(a1 + 232), v23 + 1, 8u, a5, a6);
          v23 = *(unsigned int *)(a1 + 224);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 216) + 8 * v23) = v19;
        ++*(_DWORD *)(a1 + 224);
      }
    }
  }
  return v9;
}
