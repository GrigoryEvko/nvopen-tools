// Function: sub_FCD310
// Address: 0xfcd310
//
__int64 __fastcall sub_FCD310(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rbx
  __int64 result; // rax

  if ( a3 != a4 )
  {
    v5 = a3;
    do
    {
      result = sub_FCD2E0(a1, a2, v5);
      v5 = *(_QWORD *)(v5 + 8);
    }
    while ( a4 != v5 );
  }
  return result;
}
