// Function: sub_3580F30
// Address: 0x3580f30
//
__int64 __fastcall sub_3580F30(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  if ( (_DWORD)qword_503F128 == -1 )
    return sub_357FB50(a2, (__int64)a2, a3, a4, a5, a6);
  a3 = (unsigned int)dword_503F090;
  a4 = (unsigned int)++dword_503F090;
  if ( (_DWORD)a3 == (_DWORD)qword_503F128 )
    return sub_357FB50(a2, (__int64)a2, a3, a4, a5, a6);
  else
    return 0;
}
