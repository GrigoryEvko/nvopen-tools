// Function: sub_2052A80
// Address: 0x2052a80
//
char __fastcall sub_2052A80(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v4; // rax
  __int64 *v5; // r9
  int v6; // edx
  unsigned int v7; // eax
  __int64 *v8; // rdi
  __int64 v9; // r10
  char result; // al
  int v11; // edi
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rcx
  __int64 *v15; // rdx
  int v16; // r8d
  unsigned int v17; // ecx
  __int64 *v18; // rax
  __int64 v19; // r9
  int v20; // r11d
  int v21; // eax
  int v22; // r10d

  v2 = *(_QWORD *)(a1 + 16);
  v4 = *(unsigned int *)(a1 + 32);
  v5 = (__int64 *)(v2 + 24 * v4);
  if ( (_DWORD)v4 )
  {
    v6 = v4 - 1;
    v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v2 + 24LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
    {
LABEL_3:
      if ( v5 != v8 )
        return 1;
    }
    else
    {
      v11 = 1;
      while ( v9 != -8 )
      {
        v20 = v11 + 1;
        v7 = v6 & (v11 + v7);
        v8 = (__int64 *)(v2 + 24LL * v7);
        v9 = *v8;
        if ( a2 == *v8 )
          goto LABEL_3;
        v11 = v20;
      }
    }
  }
  v12 = *(_QWORD *)(a1 + 712);
  v13 = *(_QWORD *)(v12 + 216);
  v14 = *(unsigned int *)(v12 + 232);
  v15 = (__int64 *)(v13 + 16 * v14);
  result = 0;
  if ( (_DWORD)v14 )
  {
    v16 = v14 - 1;
    v17 = (v14 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v18 = (__int64 *)(v13 + 16LL * v17);
    v19 = *v18;
    if ( a2 == *v18 )
    {
      return v15 != v18;
    }
    else
    {
      v21 = 1;
      while ( v19 != -8 )
      {
        v22 = v21 + 1;
        v17 = v16 & (v21 + v17);
        v18 = (__int64 *)(v13 + 16LL * v17);
        v19 = *v18;
        if ( a2 == *v18 )
          return v15 != v18;
        v21 = v22;
      }
      return 0;
    }
  }
  return result;
}
