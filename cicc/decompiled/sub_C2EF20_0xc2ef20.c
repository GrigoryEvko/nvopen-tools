// Function: sub_C2EF20
// Address: 0xc2ef20
//
bool __fastcall sub_C2EF20(__int64 a1)
{
  bool result; // al
  __int64 v2; // rdx

  result = 1;
  if ( (_DWORD)qword_4F83C08 != 1 )
  {
    result = 0;
    if ( (_DWORD)qword_4F83C08 != 2 )
    {
      v2 = *(_QWORD *)(a1 + 24);
      if ( !*(_DWORD *)(v2 + 24) )
        return (unsigned int)(*(_DWORD *)(v2 + 8) - 2) <= 1;
    }
  }
  return result;
}
