// Function: sub_E8D580
// Address: 0xe8d580
//
__int64 __fastcall sub_E8D580(_QWORD **a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v7; // rax
  char v8; // dl

  if ( (unsigned int)(*(_DWORD *)(*a1[37] + 56LL) - 28) > 1 && (v7 = sub_E8A510(a2, a3), v8) )
    return ((__int64 (__fastcall *)(_QWORD **, __int64, _QWORD))(*a1)[67])(a1, v7, a4);
  else
    return sub_E9A5E0(a1, a2, a3, a4);
}
