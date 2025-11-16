// Function: sub_16E4E00
// Address: 0x16e4e00
//
void __fastcall sub_16E4E00(__int64 a1)
{
  char v1; // r15
  unsigned __int64 v2; // rdx
  __int64 v3; // rsi
  int v4; // ecx
  int v5; // eax
  unsigned int v6; // r13d
  unsigned int v7; // ebx

  v1 = *(_BYTE *)(a1 + 95);
  if ( v1 )
  {
    *(_BYTE *)(a1 + 95) = 0;
    sub_16E4DB0(a1);
    v2 = *(unsigned int *)(a1 + 40);
    v3 = *(_QWORD *)(a1 + 32);
    v4 = *(_DWORD *)(v3 + 4 * v2 - 4);
    v5 = *(_DWORD *)(a1 + 40);
    v6 = v2 - 1;
    if ( !v4 )
    {
      if ( (_DWORD)v2 == 1 )
        goto LABEL_8;
      goto LABEL_5;
    }
    if ( v2 > 1 )
    {
      v1 = v4 == 4 || (unsigned int)(v4 - 1) <= 1;
      if ( !v1 )
        goto LABEL_5;
      if ( !*(_DWORD *)(v3 + 4 * v2 - 8) )
      {
        v6 = v5 - 2;
        if ( v5 == 2 )
          goto LABEL_8;
        goto LABEL_5;
      }
    }
    if ( (_DWORD)v2 == 1 )
      return;
    v1 = 0;
LABEL_5:
    v7 = 0;
    do
    {
      ++v7;
      sub_16E4B40(a1, "  ", 2u);
    }
    while ( v7 < v6 );
    if ( !v1 )
      return;
LABEL_8:
    sub_16E4B40(a1, "- ", 2u);
  }
}
