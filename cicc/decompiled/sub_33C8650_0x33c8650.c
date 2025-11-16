// Function: sub_33C8650
// Address: 0x33c8650
//
__int64 __fastcall sub_33C8650(_QWORD *a1)
{
  void (__fastcall *v1)(_QWORD *, _QWORD *, __int64); // rax
  __int64 result; // rax

  *a1 = &unk_4A36708;
  v1 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[5];
  if ( v1 )
    v1(a1 + 3, a1 + 3, 3);
  result = a1[2];
  *(_QWORD *)(result + 768) = a1[1];
  return result;
}
