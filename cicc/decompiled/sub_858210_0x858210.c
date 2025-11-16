// Function: sub_858210
// Address: 0x858210
//
__int64 __fastcall sub_858210(__int64 a1, unsigned int *a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  unsigned __int16 v6; // ax
  bool v7; // cf
  bool v8; // zf
  unsigned int *v10; // r12
  char v11; // al
  bool v12; // cf
  bool v13; // zf
  bool v14; // cf
  bool v15; // zf
  bool v16; // cf
  bool v17; // zf
  char v18; // al
  bool v19; // cf
  bool v20; // zf
  int v21; // eax

  sub_7C9660(a1);
  v6 = word_4F06418[0];
  v7 = word_4F06418[0] == 0;
  v8 = word_4F06418[0] == 1;
  if ( word_4F06418[0] != 1 )
  {
LABEL_2:
    if ( v6 == 9 )
      return sub_7C96B0(0, a2, v2, v3, v4, v5);
    goto LABEL_3;
  }
  v3 = 6;
  a1 = (__int64)"begin";
  v10 = *(unsigned int **)(qword_4D04A00 + 8);
  a2 = v10;
  do
  {
    if ( !v3 )
      break;
    v7 = *(_BYTE *)a2 < *(_BYTE *)a1;
    v8 = *(_BYTE *)a2 == *(_BYTE *)a1;
    a2 = (unsigned int *)((char *)a2 + 1);
    ++a1;
    --v3;
  }
  while ( v8 );
  v11 = (!v7 && !v8) - v7;
  v12 = 0;
  v13 = v11 == 0;
  if ( !v11 )
    goto LABEL_32;
  v3 = 4;
  a1 = (__int64)"end";
  a2 = *(unsigned int **)(qword_4D04A00 + 8);
  do
  {
    if ( !v3 )
      break;
    v12 = *(_BYTE *)a2 < *(_BYTE *)a1;
    v13 = *(_BYTE *)a2 == *(_BYTE *)a1;
    a2 = (unsigned int *)((char *)a2 + 1);
    ++a1;
    --v3;
  }
  while ( v13 );
  if ( (!v12 && !v13) == v12 )
  {
LABEL_32:
    sub_7B8B50(a1, a2, v2, v3, v4, v5);
    v6 = word_4F06418[0];
    v14 = word_4F06418[0] == 0;
    v15 = word_4F06418[0] == 1;
    if ( word_4F06418[0] != 1 )
      goto LABEL_2;
    v3 = 8;
    a1 = (__int64)"declare";
    a2 = *(unsigned int **)(qword_4D04A00 + 8);
    do
    {
      if ( !v3 )
        break;
      v14 = *(_BYTE *)a2 < *(_BYTE *)a1;
      v15 = *(_BYTE *)a2 == *(_BYTE *)a1;
      a2 = (unsigned int *)((char *)a2 + 1);
      ++a1;
      --v3;
    }
    while ( v15 );
    if ( (!v14 && !v15) == v14 )
    {
      sub_7B8B50(a1, a2, v2, v3, v4, v5);
      v6 = word_4F06418[0];
      v16 = word_4F06418[0] == 0;
      v17 = word_4F06418[0] == 1;
      if ( word_4F06418[0] != 1 )
        goto LABEL_2;
      v3 = 8;
      a1 = (__int64)"variant";
      a2 = *(unsigned int **)(qword_4D04A00 + 8);
      do
      {
        if ( !v3 )
          break;
        v16 = *(_BYTE *)a2 < *(_BYTE *)a1;
        v17 = *(_BYTE *)a2 == *(_BYTE *)a1;
        a2 = (unsigned int *)((char *)a2 + 1);
        ++a1;
        --v3;
      }
      while ( v17 );
      v18 = (!v16 && !v17) - v16;
      v19 = 0;
      v20 = v18 == 0;
      if ( !v18 )
      {
        a2 = v10;
        v3 = 6;
        a1 = (__int64)"begin";
        do
        {
          if ( !v3 )
            break;
          v19 = *(_BYTE *)a2 < *(_BYTE *)a1;
          v20 = *(_BYTE *)a2 == *(_BYTE *)a1;
          a2 = (unsigned int *)((char *)a2 + 1);
          ++a1;
          --v3;
        }
        while ( v20 );
        v21 = unk_4D03A10;
        LOBYTE(v2) = (!v19 && !v20) - v19;
        if ( (_BYTE)v2 )
        {
          --unk_4D03A10;
          if ( v21 - 1 < 0 )
          {
            a2 = &dword_4F063F8;
            a1 = 3710;
            sub_6851C0(0xE7Eu, &dword_4F063F8);
            unk_4D03A10 = 0;
            v6 = word_4F06418[0];
            goto LABEL_2;
          }
        }
        else
        {
          ++unk_4D03A10;
        }
      }
    }
  }
  do
LABEL_3:
    sub_7B8B50(a1, a2, v2, v3, v4, v5);
  while ( word_4F06418[0] != 9 );
  return sub_7C96B0(0, a2, v2, v3, v4, v5);
}
