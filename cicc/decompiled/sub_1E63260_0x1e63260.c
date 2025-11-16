// Function: sub_1E63260
// Address: 0x1e63260
//
__int64 __fastcall sub_1E63260(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r8
  unsigned int v5; // ecx
  __int64 *v6; // rdx
  __int64 v7; // r10
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rsi
  unsigned int v12; // ecx
  __int64 *v13; // rdx
  __int64 v14; // r8
  int v16; // edx
  int v17; // edx
  int v18; // r10d
  int v19; // ebx

  v3 = *(unsigned int *)(a3 + 24);
  if ( !(_DWORD)v3 )
    return a2[1];
  v4 = *(_QWORD *)(a3 + 8);
  v5 = (v3 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v6 = (__int64 *)(v4 + 16LL * v5);
  v7 = *v6;
  if ( *a2 == *v6 )
  {
LABEL_3:
    if ( v6 != (__int64 *)(v4 + 16 * v3) )
    {
      v8 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 232LL);
      v9 = *(unsigned int *)(v8 + 72);
      if ( (_DWORD)v9 )
      {
        v10 = v6[1];
        v11 = *(_QWORD *)(v8 + 56);
        v12 = (v9 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v13 = (__int64 *)(v11 + 16LL * v12);
        v14 = *v13;
        if ( v10 == *v13 )
        {
LABEL_6:
          if ( v13 != (__int64 *)(v11 + 16 * v9) )
            return *(_QWORD *)(v13[1] + 8);
        }
        else
        {
          v17 = 1;
          while ( v14 != -8 )
          {
            v18 = v17 + 1;
            v12 = (v9 - 1) & (v17 + v12);
            v13 = (__int64 *)(v11 + 16LL * v12);
            v14 = *v13;
            if ( v10 == *v13 )
              goto LABEL_6;
            v17 = v18;
          }
        }
      }
      BUG();
    }
  }
  else
  {
    v16 = 1;
    while ( v7 != -8 )
    {
      v19 = v16 + 1;
      v5 = (v3 - 1) & (v16 + v5);
      v6 = (__int64 *)(v4 + 16LL * v5);
      v7 = *v6;
      if ( *a2 == *v6 )
        goto LABEL_3;
      v16 = v19;
    }
  }
  return a2[1];
}
