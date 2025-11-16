// Function: sub_392A800
// Address: 0x392a800
//
__int64 __fastcall sub_392A800(__int64 a1, __int64 a2)
{
  char *v3; // rdi
  char v4; // dl
  char *v5; // rax
  _BYTE *v6; // rcx
  char v7; // al
  _BYTE *v8; // rax
  __int64 v9; // rax
  _BYTE *v10; // rcx

  v3 = *(char **)(a2 + 144);
  v4 = *v3;
  if ( (unsigned __int8)(*v3 - 48) <= 9u )
  {
    v5 = v3 + 1;
    do
    {
      *(_QWORD *)(a2 + 144) = v5;
      v4 = *v5;
      v3 = v5++;
    }
    while ( (unsigned __int8)(v4 - 48) <= 9u );
  }
  if ( (v4 & 0xDF) == 0x45 )
  {
    v6 = v3 + 1;
    *(_QWORD *)(a2 + 144) = v3 + 1;
    v7 = v3[1];
    if ( ((v7 - 43) & 0xFD) == 0 )
    {
      v6 = v3 + 2;
      *(_QWORD *)(a2 + 144) = v3 + 2;
      v7 = v3[2];
    }
    if ( (unsigned __int8)(v7 - 48) <= 9u )
    {
      v8 = v6 + 1;
      do
      {
        v6 = v8;
        *(_QWORD *)(a2 + 144) = v8++;
      }
      while ( (unsigned __int8)(*v6 - 48) <= 9u );
    }
  }
  else
  {
    v6 = *(_BYTE **)(a2 + 144);
  }
  v9 = *(_QWORD *)(a2 + 104);
  *(_DWORD *)a1 = 6;
  *(_DWORD *)(a1 + 32) = 64;
  v10 = &v6[-v9];
  *(_QWORD *)(a1 + 8) = v9;
  *(_QWORD *)(a1 + 16) = v10;
  *(_QWORD *)(a1 + 24) = 0;
  return a1;
}
