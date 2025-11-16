// Function: sub_8CBA30
// Address: 0x8cba30
//
__int64 **__fastcall sub_8CBA30(__int64 a1)
{
  __int64 **result; // rax

  result = *(__int64 ***)(a1 + 32);
  if ( result && qword_4F074B0 == qword_4F60258 )
  {
    if ( *((char *)*result + 162) < 0 )
      sub_8C6700(*result, (unsigned int *)(a1 + 64), 0x42Au, 0x425u);
    sub_8CB6C0(6u, a1);
    return (__int64 **)sub_8CA420(a1);
  }
  return result;
}
