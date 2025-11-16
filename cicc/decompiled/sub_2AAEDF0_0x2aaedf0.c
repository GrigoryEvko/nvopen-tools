// Function: sub_2AAEDF0
// Address: 0x2aaedf0
//
__int64 __fastcall sub_2AAEDF0(__int64 a1, __int64 a2)
{
  if ( ((*(_BYTE *)(a1 + 8) - 7) & 0xFD) != 0 && (BYTE4(a2) || (_DWORD)a2 != 1) )
    return sub_BCE1B0((__int64 *)a1, a2);
  else
    return a1;
}
