// Function: sub_A4FAF0
// Address: 0xa4faf0
//
bool __fastcall sub_A4FAF0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r11
  __int64 v5; // rcx
  __int64 v6; // rax
  __int64 v7; // r8
  int v8; // r9d
  unsigned int v9; // r10d
  __int64 *v10; // rdx
  __int64 v11; // rbx
  __int64 *v12; // rax
  __int64 v13; // rbx
  unsigned int v14; // r10d
  unsigned int v15; // r13d
  __int64 *v16; // r11
  __int64 v17; // rdx
  char *v18; // rdx
  unsigned int v19; // eax
  char v21; // dl
  unsigned int v22; // ebx
  int v23; // r11d
  int v24; // r14d
  int v25; // edx
  unsigned int v26; // ebx
  int v27; // r13d

  if ( a2 == a3 )
    return 0;
  v3 = *(_QWORD *)(a2 + 24);
  v5 = *a1;
  v6 = *(unsigned int *)(*a1 + 24);
  v7 = *(_QWORD *)(*a1 + 8);
  if ( !(_DWORD)v6 )
  {
    v14 = 0;
LABEL_14:
    v18 = (char *)a1[1];
    goto LABEL_15;
  }
  v8 = v6 - 1;
  v9 = (v6 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( v3 == *v10 )
  {
LABEL_4:
    v12 = (__int64 *)(v7 + 16 * v6);
    if ( v10 != v12 )
    {
      v13 = *(_QWORD *)(a3 + 24);
      v14 = *(_DWORD *)(*(_QWORD *)(v5 + 32) + 16LL * *((unsigned int *)v10 + 2) + 8);
      goto LABEL_6;
    }
  }
  else
  {
    v25 = 1;
    while ( v11 != -4096 )
    {
      v27 = v25 + 1;
      v9 = v8 & (v25 + v9);
      v10 = (__int64 *)(v7 + 16LL * v9);
      v11 = *v10;
      if ( v3 == *v10 )
        goto LABEL_4;
      v25 = v27;
    }
    v12 = (__int64 *)(v7 + 16 * v6);
  }
  v13 = *(_QWORD *)(a3 + 24);
  v14 = 0;
LABEL_6:
  v15 = v8 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
  v16 = (__int64 *)(v7 + 16LL * v15);
  v17 = *v16;
  if ( *v16 != v13 )
  {
    v23 = 1;
    while ( v17 != -4096 )
    {
      v24 = v23 + 1;
      v15 = v8 & (v15 + v23);
      v16 = (__int64 *)(v7 + 16LL * v15);
      v17 = *v16;
      if ( *v16 == v13 )
        goto LABEL_7;
      v23 = v24;
    }
    goto LABEL_14;
  }
LABEL_7:
  v18 = (char *)a1[1];
  if ( v16 == v12 )
  {
LABEL_15:
    v19 = 0;
    goto LABEL_16;
  }
  v19 = *(_DWORD *)(*(_QWORD *)(v5 + 32) + 16LL * *((unsigned int *)v16 + 2) + 8);
  if ( v19 > v14 )
    return *v18 && *(_DWORD *)a1[2] >= v19;
LABEL_16:
  v21 = *v18;
  if ( v14 > v19 )
    return !v21 || *(_DWORD *)a1[2] < v14;
  if ( v21 && *(_DWORD *)a1[2] >= v14 )
  {
    v26 = sub_BD2910(a2);
    return v26 < (unsigned int)sub_BD2910(a3);
  }
  else
  {
    v22 = sub_BD2910(a2);
    return v22 > (unsigned int)sub_BD2910(a3);
  }
}
