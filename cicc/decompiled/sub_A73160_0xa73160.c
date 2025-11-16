// Function: sub_A73160
// Address: 0xa73160
//
__int64 __fastcall sub_A73160(__int64 a1)
{
  __int64 result; // rax

  result = 0;
  if ( *(_QWORD *)a1 )
    return *(unsigned int *)(*(_QWORD *)a1 + 8LL);
  return result;
}
