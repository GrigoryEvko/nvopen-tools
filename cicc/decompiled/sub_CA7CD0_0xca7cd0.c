// Function: sub_CA7CD0
// Address: 0xca7cd0
//
__int64 __fastcall sub_CA7CD0(__int64 a1, char *a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // r14
  __int64 (__fastcall *v6)(_QWORD *, __int64); // rax
  __int64 v7; // rax

  v4 = (_QWORD *)(a1 + a3);
  while ( 1 )
  {
    v6 = (__int64 (__fastcall *)(_QWORD *, __int64))a2;
    if ( ((unsigned __int8)a2 & 1) != 0 )
      v6 = *(__int64 (__fastcall **)(_QWORD *, __int64))&a2[*v4 - 1];
    v7 = v6(v4, a4);
    if ( a4 == v7 )
      break;
    a4 = v7;
  }
  return a4;
}
