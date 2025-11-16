// Function: sub_947060
// Address: 0x947060
//
__int64 __fastcall sub_947060(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v6; // rdi
  char v7; // cl
  __int64 v8; // rax
  char v9; // dl
  __int64 v10; // rax
  int v11; // eax

  v6 = a3;
  v7 = *(_BYTE *)(a3 + 140);
  if ( v7 == 12 )
  {
    v8 = a3;
    do
    {
      v8 = *(_QWORD *)(v8 + 160);
      v9 = *(_BYTE *)(v8 + 140);
    }
    while ( v9 == 12 );
  }
  else
  {
    v9 = *(_BYTE *)(a3 + 140);
  }
  if ( !unk_4D04638 && (unsigned __int8)(v9 - 9) <= 2u )
  {
    v10 = v6;
    if ( v7 == 12 )
    {
      do
        v10 = *(_QWORD *)(v10 + 160);
      while ( *(_BYTE *)(v10 + 140) == 12 );
    }
    if ( *(_QWORD *)(v10 + 128) > (unsigned __int64)dword_4D04634 )
    {
      if ( *(char *)(v6 + 142) < 0 )
        goto LABEL_11;
      goto LABEL_10;
    }
  }
  if ( a4 )
  {
    if ( *(char *)(v6 + 142) < 0 )
      goto LABEL_11;
LABEL_10:
    if ( v7 == 12 )
    {
      v11 = sub_8D4AB0(v6);
      goto LABEL_12;
    }
LABEL_11:
    v11 = *(_DWORD *)(v6 + 136);
LABEL_12:
    *(_DWORD *)(a1 + 8) = v11;
    *(_QWORD *)a1 = 0;
    *(_DWORD *)(a1 + 12) = 2;
    *(_BYTE *)(a1 + 16) = 1;
    return a1;
  }
  while ( v7 == 12 )
  {
    v6 = *(_QWORD *)(v6 + 160);
    v7 = *(_BYTE *)(v6 + 140);
  }
  if ( v7 == 2 && *(_BYTE *)(v6 + 160) <= 4u )
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0x100000000LL;
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
  else
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
}
