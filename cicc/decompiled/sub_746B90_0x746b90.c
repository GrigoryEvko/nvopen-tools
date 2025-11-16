// Function: sub_746B90
// Address: 0x746b90
//
__int64 __fastcall sub_746B90(_DWORD *a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  __int64 v3; // rdx

  result = qword_4F08008;
  if ( qword_4F08008 )
  {
    v2 = (int)a1[1];
    if ( (int)v2 <= dword_4F08000 && (_DWORD)v2 )
    {
      v3 = qword_4F08008 + 16 * v2 - 16;
      result = 0;
      if ( *a1 <= *(_DWORD *)v3 )
        return *(_QWORD *)(*(_QWORD *)(v3 + 8) + 8LL * (unsigned int)(*a1 - 1));
    }
    else
    {
      return 0;
    }
  }
  return result;
}
