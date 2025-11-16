// Function: sub_16BDD10
// Address: 0x16bdd10
//
__int64 __fastcall sub_16BDD10(_QWORD *a1, __int64 *a2)
{
  __int64 result; // rax

  while ( 1 )
  {
    result = *a2;
    if ( *a2 == -1 || result && (result & 1) == 0 )
      break;
    ++a2;
  }
  *a1 = result;
  return result;
}
