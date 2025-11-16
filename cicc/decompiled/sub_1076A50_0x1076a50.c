// Function: sub_1076A50
// Address: 0x1076a50
//
__int64 __fastcall sub_1076A50(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdi

  *a1 = &unk_49E6108;
  v3 = a1[15];
  if ( v3 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 8LL))(v3);
  sub_E8EC10((__int64)a1, a2);
  return j_j___libc_free_0(a1, 144);
}
