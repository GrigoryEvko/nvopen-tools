// Function: sub_C8B430
// Address: 0xc8b430
//
__int64 (*sub_C8B430())(void)
{
  int *v0; // rax
  int v1; // r12d
  int *v2; // rbx
  __int64 (*result)(void); // rax

  v0 = __errno_location();
  v1 = *v0;
  v2 = v0;
  result = qword_4F84BD0;
  if ( qword_4F84BD0 )
    result = (__int64 (*)(void))qword_4F84BD0();
  *v2 = v1;
  return result;
}
