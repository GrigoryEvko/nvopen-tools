// Function: sub_2E97E80
// Address: 0x2e97e80
//
__int64 __fastcall sub_2E97E80(__int64 a1, __int64 a2, char a3)
{
  char v5; // al
  unsigned int *v6; // rax
  unsigned int *v7; // rcx
  int v8; // r8d
  _QWORD *v9; // rdx
  _QWORD *v10; // rdi
  __int64 v12; // rdx
  __int64 v13; // rax

  v5 = *(_BYTE *)(a2 + 8) & 1;
  if ( *(_DWORD *)(a2 + 8) >> 1 )
  {
    if ( v5 )
    {
      v6 = (unsigned int *)(a2 + 16);
      v7 = (unsigned int *)(a2 + 48);
    }
    else
    {
      v6 = *(unsigned int **)(a2 + 16);
      v7 = &v6[2 * *(unsigned int *)(a2 + 24)];
      if ( v6 == v7 )
        return 0;
    }
    do
    {
      if ( *v6 <= 0xFFFFFFFD )
        break;
      v6 += 2;
    }
    while ( v7 != v6 );
  }
  else
  {
    if ( v5 )
    {
      v12 = a2 + 16;
      v13 = 32;
    }
    else
    {
      v12 = *(_QWORD *)(a2 + 16);
      v13 = 8LL * *(unsigned int *)(a2 + 24);
    }
    v6 = (unsigned int *)(v12 + v13);
    v7 = v6;
  }
  if ( v6 != v7 )
  {
LABEL_7:
    v8 = v6[1];
    if ( v8 <= 0 )
      goto LABEL_16;
    if ( !a3 || byte_5020788 )
    {
      v9 = *(_QWORD **)(a1 + 640);
      v10 = &v9[6 * *(unsigned int *)(a1 + 648)];
      if ( v10 == v9 )
      {
LABEL_16:
        while ( 1 )
        {
          v6 += 2;
          if ( v6 == v7 )
            return 0;
          if ( *v6 <= 0xFFFFFFFD )
          {
            if ( v7 != v6 )
              goto LABEL_7;
            return 0;
          }
        }
      }
      while ( v8 + *(_DWORD *)(*v9 + 4LL * *v6) < *(_DWORD *)(*(_QWORD *)(a1 + 592) + 4LL * *v6) )
      {
        v9 += 6;
        if ( v10 == v9 )
          goto LABEL_16;
      }
    }
    return 1;
  }
  return 0;
}
