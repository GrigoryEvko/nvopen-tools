// Function: sub_1443D90
// Address: 0x1443d90
//
__int64 __fastcall sub_1443D90(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // rax
  int v4; // r11d
  __int64 v5; // rcx
  unsigned int v6; // ebx
  __int64 *v7; // rdx
  __int64 v8; // r9
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rsi
  unsigned int v13; // ecx
  __int64 *v14; // rdx
  __int64 v15; // r8
  int v17; // edx
  int v18; // r10d

  v3 = *(unsigned int *)(a3 + 24);
  if ( !(_DWORD)v3 )
    return a2[1];
  v4 = 1;
  v5 = *(_QWORD *)(a3 + 8);
  v6 = (v3 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v7 = (__int64 *)(v5 + 16LL * v6);
  v8 = *v7;
  if ( *a2 == *v7 )
  {
LABEL_3:
    if ( v7 != (__int64 *)(v5 + 16 * v3) )
    {
      v9 = *(_QWORD *)(a1 + 16);
      v10 = *(unsigned int *)(v9 + 72);
      if ( (_DWORD)v10 )
      {
        v11 = v7[1];
        v12 = *(_QWORD *)(v9 + 56);
        v13 = (v10 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v14 = (__int64 *)(v12 + 16LL * v13);
        v15 = *v14;
        if ( v11 == *v14 )
        {
LABEL_6:
          if ( v14 != (__int64 *)(v12 + 16 * v10) )
            return *(_QWORD *)(v14[1] + 8);
        }
        else
        {
          v17 = 1;
          while ( v15 != -8 )
          {
            v18 = v17 + 1;
            v13 = (v10 - 1) & (v17 + v13);
            v14 = (__int64 *)(v12 + 16LL * v13);
            v15 = *v14;
            if ( v11 == *v14 )
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
    while ( v8 != -8 )
    {
      v6 = (v3 - 1) & (v6 + v4);
      v7 = (__int64 *)(v5 + 16LL * v6);
      v8 = *v7;
      if ( *a2 == *v7 )
        goto LABEL_3;
      ++v4;
    }
  }
  return a2[1];
}
