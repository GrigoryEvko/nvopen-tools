// Function: sub_86A500
// Address: 0x86a500
//
__int64 *__fastcall sub_86A500(__int64 *a1)
{
  __int64 *result; // rax

  for ( result = a1; *((_BYTE *)result + 16) != 54 || *(_QWORD *)(result[3] + 16) != a1[3]; result = (__int64 *)*result )
    ;
  return result;
}
