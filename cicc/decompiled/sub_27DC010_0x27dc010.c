// Function: sub_27DC010
// Address: 0x27dc010
//
__int64 __fastcall sub_27DC010(__int64 a1)
{
  unsigned __int64 v1; // rax
  int v2; // edx
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rdx
  char v6; // cl
  __int64 v7; // rax
  int v8; // ecx
  unsigned int v9; // r12d
  int v10; // r13d
  unsigned int v11; // r15d
  unsigned int v12; // r14d
  __int64 v13; // rdx
  char v14; // cl
  __int64 v15; // rax
  int v16; // ecx
  unsigned int v17; // eax

  v1 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v1 == a1 + 48 )
  {
    v3 = 0;
  }
  else
  {
    if ( !v1 )
      BUG();
    v2 = *(unsigned __int8 *)(v1 - 24);
    v3 = 0;
    v4 = v1 - 24;
    if ( (unsigned int)(v2 - 30) < 0xB )
      v3 = v4;
  }
  v5 = *(_QWORD *)(sub_B46EC0(v3, 0) + 16);
  while ( v5 )
  {
    v6 = **(_BYTE **)(v5 + 24);
    v7 = v5;
    v5 = *(_QWORD *)(v5 + 8);
    if ( (unsigned __int8)(v6 - 30) <= 0xAu )
    {
      v8 = 0;
      while ( 1 )
      {
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
          break;
        while ( (unsigned __int8)(**(_BYTE **)(v7 + 24) - 30) <= 0xAu )
        {
          v7 = *(_QWORD *)(v7 + 8);
          ++v8;
          if ( !v7 )
            goto LABEL_12;
        }
      }
LABEL_12:
      v9 = v8 + 1;
      v10 = sub_B46E30(v3);
      if ( v10 != 1 )
        goto LABEL_13;
      return 0;
    }
  }
  v9 = 0;
  v10 = sub_B46E30(v3);
  if ( v10 == 1 )
    return 0;
LABEL_13:
  v11 = 1;
  v12 = 0;
  do
  {
    v13 = *(_QWORD *)(sub_B46EC0(v3, v11) + 16);
    do
    {
      if ( !v13 )
      {
        v17 = 0;
        goto LABEL_22;
      }
      v14 = **(_BYTE **)(v13 + 24);
      v15 = v13;
      v13 = *(_QWORD *)(v13 + 8);
    }
    while ( (unsigned __int8)(v14 - 30) > 0xAu );
    v16 = 0;
    while ( 1 )
    {
      v15 = *(_QWORD *)(v15 + 8);
      if ( !v15 )
        break;
      while ( (unsigned __int8)(**(_BYTE **)(v15 + 24) - 30) <= 0xAu )
      {
        v15 = *(_QWORD *)(v15 + 8);
        ++v16;
        if ( !v15 )
          goto LABEL_21;
      }
    }
LABEL_21:
    v17 = v16 + 1;
LABEL_22:
    if ( v17 < v9 )
    {
      v9 = v17;
      v12 = v11;
    }
    ++v11;
  }
  while ( v10 != v11 );
  return v12;
}
