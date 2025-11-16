// Function: sub_2C4FB60
// Address: 0x2c4fb60
//
_QWORD *__fastcall sub_2C4FB60(_QWORD *a1, int a2)
{
  _BYTE *v2; // rax
  unsigned int v4; // ecx
  char v5; // dl
  _BYTE *v6; // rdi

  while ( 1 )
  {
    v2 = (_BYTE *)*a1;
    if ( *(_BYTE *)*a1 != 92 )
      return a1;
    a2 = *(_DWORD *)(*((_QWORD *)v2 + 9) + 4LL * (unsigned int)a2);
    v4 = *(_DWORD *)(*(_QWORD *)(*((_QWORD *)v2 - 8) + 8LL) + 32LL);
    if ( a2 < 0 )
      break;
    v5 = v2[7] & 0x40;
    if ( a2 >= v4 )
    {
      if ( v5 )
        v6 = (_BYTE *)*((_QWORD *)v2 - 1);
      else
        v6 = &v2[-32 * (*((_DWORD *)v2 + 1) & 0x7FFFFFF)];
      a1 = v6 + 32;
      a2 -= v4;
    }
    else if ( v5 )
    {
      a1 = (_QWORD *)*((_QWORD *)v2 - 1);
    }
    else
    {
      a1 = &v2[-32 * (*((_DWORD *)v2 + 1) & 0x7FFFFFF)];
    }
  }
  return 0;
}
