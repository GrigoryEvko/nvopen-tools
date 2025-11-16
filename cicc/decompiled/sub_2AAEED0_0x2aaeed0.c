// Function: sub_2AAEED0
// Address: 0x2aaeed0
//
_QWORD *__fastcall sub_2AAEED0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v6; // al

  v6 = *(_BYTE *)(a1 + 8);
  if ( v6 == 15 )
    return sub_E454C0(a1, a2, a3, a4, a5, a6);
  if ( ((v6 - 7) & 0xFD) != 0 && (BYTE4(a2) || (_DWORD)a2 != 1) )
    return (_QWORD *)sub_BCE1B0((__int64 *)a1, a2);
  return (_QWORD *)a1;
}
