// Function: sub_131C450
// Address: 0x131c450
//
_QWORD *__fastcall sub_131C450(_BYTE *a1, __int64 a2)
{
  _QWORD *result; // rax
  __int64 v3; // [rsp+8h] [rbp-8h] BYREF

  result = sub_131C150(a1, a2, 128, 128, &v3);
  if ( result )
    result[2] = v3 & 0xFFF | result[2] & 0xFFFFFFFFFFFFF000LL;
  return result;
}
