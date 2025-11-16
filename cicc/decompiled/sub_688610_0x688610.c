// Function: sub_688610
// Address: 0x688610
//
_BOOL8 __fastcall sub_688610(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r13
  __int64 v7; // rsi
  _BOOL8 result; // rax
  int v9; // ebx

  v6 = *a1;
  v7 = *a2;
  if ( *((_BYTE *)a1 + 17) == 1 && *((_BYTE *)a2 + 17) == 1 )
  {
    v9 = ((__int64 (*)(void))sub_6ED0A0)();
    if ( v9 == (unsigned int)sub_6ED0A0(a2) && (!dword_4F077BC || (_DWORD)qword_4F077B4 || qword_4F077A8 > 0xC3B3u) )
    {
      while ( *(_BYTE *)(v6 + 140) == 12 )
        v6 = *(_QWORD *)(v6 + 160);
      while ( *(_BYTE *)(v7 + 140) == 12 )
        v7 = *(_QWORD *)(v7 + 160);
    }
  }
  result = 1;
  if ( v6 != v7 )
    return (unsigned int)sub_8D97D0(v6, v7, 0, a4, a5) != 0;
  return result;
}
