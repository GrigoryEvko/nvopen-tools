// Function: sub_31D6700
// Address: 0x31d6700
//
__int64 __fastcall sub_31D6700(__int64 a1)
{
  __int64 v2; // r14
  unsigned int v3; // eax
  unsigned int v4; // r12d
  __int64 v5; // r13
  unsigned __int64 i; // r14
  int v7; // eax
  __int64 v8; // rax
  int v9; // eax
  __int64 v10; // rax
  unsigned __int64 j; // rax
  __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 v16; // rdx

  v2 = **(_QWORD **)(a1 + 64);
  LOBYTE(v3) = sub_2E322F0(v2, a1);
  if ( !(_BYTE)v3 )
    return 0;
  v4 = v3;
  v5 = v2 + 48;
  if ( v2 + 48 != (*(_QWORD *)(v2 + 48) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    for ( i = sub_2E313E0(v2); v5 != i; i = *(_QWORD *)(i + 8) )
    {
      v7 = *(_DWORD *)(i + 44);
      if ( (v7 & 4) == 0 && (v7 & 8) != 0 )
        LOBYTE(v8) = sub_2E88A90(i, 1024, 1);
      else
        v8 = (*(_QWORD *)(*(_QWORD *)(i + 16) + 24LL) >> 10) & 1LL;
      if ( !(_BYTE)v8 )
        return 0;
      v9 = *(_DWORD *)(i + 44);
      if ( (v9 & 4) == 0 && (v9 & 8) != 0 )
        LOBYTE(v10) = sub_2E88A90(i, 2048, 1);
      else
        v10 = (*(_QWORD *)(*(_QWORD *)(i + 16) + 24LL) >> 11) & 1LL;
      if ( (_BYTE)v10 )
        return 0;
      for ( j = i; (*(_BYTE *)(j + 44) & 4) != 0; j = *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL )
        ;
      v12 = *(_QWORD *)(i + 24) + 48LL;
      do
      {
        v13 = *(_QWORD *)(j + 32);
        v14 = v13 + 40LL * (*(_DWORD *)(j + 40) & 0xFFFFFF);
        if ( v13 != v14 )
          goto LABEL_18;
        j = *(_QWORD *)(j + 8);
        if ( v12 == j )
          goto LABEL_27;
      }
      while ( (*(_BYTE *)(j + 44) & 4) != 0 );
      j = *(_QWORD *)(i + 24) + 48LL;
LABEL_27:
      while ( v13 != v14 )
      {
        while ( 1 )
        {
LABEL_18:
          if ( *(_BYTE *)v13 == 8 || *(_BYTE *)v13 == 4 && a1 == *(_QWORD *)(v13 + 24) )
            return 0;
          v15 = v13 + 40;
          v16 = v14;
          if ( v15 == v14 )
            break;
          v14 = v15;
LABEL_34:
          v13 = v14;
          v14 = v16;
        }
        while ( 1 )
        {
          j = *(_QWORD *)(j + 8);
          if ( v12 == j )
            break;
          if ( (*(_BYTE *)(j + 44) & 4) == 0 )
          {
            j = *(_QWORD *)(i + 24) + 48LL;
            break;
          }
          v14 = *(_QWORD *)(j + 32);
          v16 = v14 + 40LL * (*(_DWORD *)(j + 40) & 0xFFFFFF);
          if ( v14 != v16 )
            goto LABEL_34;
        }
        v13 = v14;
        v14 = v16;
      }
      if ( (*(_BYTE *)i & 4) == 0 && (*(_DWORD *)(i + 44) & 8) != 0 )
      {
        do
          i = *(_QWORD *)(i + 8);
        while ( (*(_BYTE *)(i + 44) & 8) != 0 );
      }
    }
  }
  return v4;
}
