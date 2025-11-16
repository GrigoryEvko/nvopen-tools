// Function: sub_85B940
// Address: 0x85b940
//
__int64 __fastcall sub_85B940(__int64 *a1)
{
  __int64 result; // rax
  __int64 *i; // rbx

  result = sub_85B1E0();
  if ( !(_DWORD)result )
  {
    result = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (*(_BYTE *)(result + 12) & 4) == 0 )
    {
      for ( i = a1; i; i = (__int64 *)*i )
        result = sub_6851A0(0x77Fu, (_DWORD *)i + 5, *(_QWORD *)(*(_QWORD *)i[1] + 8LL));
    }
  }
  return result;
}
