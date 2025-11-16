// Function: sub_2B15E10
// Address: 0x2b15e10
//
__int64 __fastcall sub_2B15E10(char *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  unsigned __int8 v5; // al
  unsigned __int8 **v7; // rdi

  v5 = *a1;
  if ( (unsigned __int8)*a1 <= 0x1Cu )
  {
    LOBYTE(a5) = (unsigned __int8)(v5 - 12) <= 1u;
    return a5;
  }
  LOBYTE(a5) = v5 == 93 || (unsigned __int8)(v5 - 90) <= 1u;
  if ( !(_BYTE)a5 || v5 == 93 )
    return a5;
  if ( (a1[7] & 0x40) != 0 )
  {
    v7 = (unsigned __int8 **)*((_QWORD *)a1 - 1);
    if ( *(_BYTE *)(*((_QWORD *)*v7 + 1) + 8LL) == 17 )
      goto LABEL_8;
    return 0;
  }
  v7 = (unsigned __int8 **)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
  if ( *(_BYTE *)(*((_QWORD *)*v7 + 1) + 8LL) != 17 )
    return 0;
LABEL_8:
  if ( v5 == 90 )
    return sub_2B0D8B0(v7[4]);
  else
    return sub_2B0D8B0(v7[8]);
}
