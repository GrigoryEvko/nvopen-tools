// Function: sub_2B14010
// Address: 0x2b14010
//
char __fastcall sub_2B14010(_QWORD *a1, int a2)
{
  __int64 v3; // rcx
  __int64 v4; // rax
  unsigned int v5; // edx
  int *v6; // rdi
  int v7; // r9d
  int *v8; // rdx
  char result; // al
  int v10; // edi
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rdx
  int *v14; // r9
  int v15; // eax
  unsigned int v16; // edx
  int *v17; // rdi
  int v18; // r8d
  int v19; // r11d
  int v20; // edi
  int v21; // r10d

  v3 = *(_QWORD *)(*a1 + 8LL);
  v4 = *(unsigned int *)(*a1 + 24LL);
  if ( (_DWORD)v4 )
  {
    v5 = (v4 - 1) & (37 * a2);
    v6 = (int *)(v3 + 4LL * v5);
    v7 = *v6;
    if ( *v6 == a2 )
    {
LABEL_3:
      v8 = (int *)(v3 + 4 * v4);
      result = 0;
      if ( v6 != v8 )
        return result;
    }
    else
    {
      v10 = 1;
      while ( v7 != -1 )
      {
        v19 = v10 + 1;
        v5 = (v4 - 1) & (v10 + v5);
        v6 = (int *)(v3 + 4LL * v5);
        v7 = *v6;
        if ( *v6 == a2 )
          goto LABEL_3;
        v10 = v19;
      }
    }
  }
  v11 = a1[1];
  v12 = *(_QWORD *)(v11 + 8);
  v13 = *(unsigned int *)(v11 + 24);
  v14 = (int *)(v12 + 4 * v13);
  if ( (_DWORD)v13 )
  {
    v15 = v13 - 1;
    v16 = (v13 - 1) & (37 * a2);
    v17 = (int *)(v12 + 4LL * v16);
    v18 = *v17;
    if ( *v17 == a2 )
      return v17 == v14;
    v20 = 1;
    while ( v18 != -1 )
    {
      v21 = v20 + 1;
      v16 = v15 & (v20 + v16);
      v17 = (int *)(v12 + 4LL * v16);
      v18 = *v17;
      if ( *v17 == a2 )
        return v17 == v14;
      v20 = v21;
    }
  }
  return 1;
}
