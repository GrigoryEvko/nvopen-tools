// Function: sub_8AC4A0
// Address: 0x8ac4a0
//
__int64 __fastcall sub_8AC4A0(__int64 *a1)
{
  __int64 v1; // r13
  __int64 v2; // r12
  __int64 result; // rax

  v1 = *a1;
  v2 = sub_892240(*a1);
  sub_892270((_QWORD *)v2);
  if ( !*qword_4D03FD0 || (result = sub_880F80(v1), qword_4D03FF0 == result) )
  {
    result = *(_QWORD *)(v2 + 16);
    if ( (*(_BYTE *)(result + 28) & 1) == 0 )
    {
      if ( *(char *)(v2 + 80) >= 0 )
        result = sub_899CC0(v2, 0, 0);
      else
        result = *(_BYTE *)(v2 + 80) >> 7;
      if ( (_DWORD)result )
        return sub_8AB5A0(v2);
    }
  }
  return result;
}
