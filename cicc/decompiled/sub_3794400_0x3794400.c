// Function: sub_3794400
// Address: 0x3794400
//
__int64 __fastcall sub_3794400(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // edx
  int v5; // edx
  __int64 v6; // r13
  int v7; // r12d
  __int64 v8; // r10
  __int64 v9; // rdi
  unsigned __int64 v10; // r8
  int v11; // r9d
  int v12; // r11d
  unsigned __int64 v13; // rax
  unsigned int i; // eax
  __int64 v15; // rsi
  unsigned int v16; // eax
  int v18; // ebx

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v5 = v4 - 1;
  v6 = 0;
  v7 = 1;
  v8 = *(_QWORD *)(a1 + 8);
  v9 = *a2;
  v11 = *((_DWORD *)a2 + 6);
  v12 = *((_DWORD *)a2 + 2);
  v10 = a2[2];
  v13 = 0xBF58476D1CE4E5B9LL
      * ((v11 + ((unsigned int)(v10 >> 9) ^ (unsigned int)(v10 >> 4)))
       | ((unsigned __int64)(v12
                           + ((unsigned int)((unsigned __int64)*a2 >> 9) ^ (unsigned int)((unsigned __int64)*a2 >> 4))) << 32));
  for ( i = v5 & ((v13 >> 31) ^ v13); ; i = v5 & v16 )
  {
    v15 = v8 + 40LL * i;
    if ( v9 == *(_QWORD *)v15
      && v12 == *(_DWORD *)(v15 + 8)
      && v10 == *(_QWORD *)(v15 + 16)
      && v11 == *(_DWORD *)(v15 + 24) )
    {
      *a3 = v15;
      return 1;
    }
    if ( !*(_QWORD *)v15 )
      break;
LABEL_5:
    v16 = v7 + i;
    ++v7;
  }
  v18 = *(_DWORD *)(v15 + 8);
  if ( v18 != -1 )
  {
    if ( v18 == -2 && !*(_QWORD *)(v15 + 16) && *(_DWORD *)(v15 + 24) == -2 && !v6 )
      v6 = v8 + 40LL * i;
    goto LABEL_5;
  }
  if ( *(_QWORD *)(v15 + 16) || *(_DWORD *)(v15 + 24) != -1 )
    goto LABEL_5;
  if ( !v6 )
    v6 = v8 + 40LL * i;
  *a3 = v6;
  return 0;
}
