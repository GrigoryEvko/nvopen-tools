// Function: sub_1076A10
// Address: 0x1076a10
//
__int64 __fastcall sub_1076A10(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdi

  *a1 = &unk_49E6108;
  v3 = a1[15];
  if ( v3 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 8LL))(v3);
  return sub_E8EC10((__int64)a1, a2);
}
