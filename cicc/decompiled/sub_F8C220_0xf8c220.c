// Function: sub_F8C220
// Address: 0xf8c220
//
_QWORD *__fastcall sub_F8C220(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // eax

  v3 = *(_DWORD *)(a2 + 32);
  if ( v3 == 1 )
    return sub_F8AD20(a1, a2, a3);
  if ( v3 == 2 )
    return (_QWORD *)sub_F8C070(a1, a2, a3);
  if ( v3 )
    BUG();
  return (_QWORD *)sub_F8C250(a1, a2, a3);
}
