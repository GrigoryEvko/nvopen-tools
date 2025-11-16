// Function: sub_C9AF60
// Address: 0xc9af60
//
__int64 *__fastcall sub_C9AF60(__int64 a1)
{
  __int64 *result; // rax
  __int64 v3; // rdi

  result = (__int64 *)sub_C94E20((__int64)&qword_4F84F00);
  v3 = qword_4F84F10;
  if ( result )
    v3 = *result;
  if ( v3 )
    return (__int64 *)sub_C9A3C0(v3, a1);
  return result;
}
