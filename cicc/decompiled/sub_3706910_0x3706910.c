// Function: sub_3706910
// Address: 0x3706910
//
__int64 *__fastcall sub_3706910(__int64 *a1, __int64 **a2, _QWORD *a3)
{
  _QWORD v5[5]; // [rsp+8h] [rbp-28h] BYREF

  (*(void (__fastcall **)(_QWORD *))(**a2 + 32))(v5);
  if ( (v5[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    *a1 = v5[0] & 0xFFFFFFFFFFFFFFFELL | 1;
  else
    sub_3705CB0(a1, a2, a3);
  return a1;
}
