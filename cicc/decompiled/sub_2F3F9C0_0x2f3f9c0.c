// Function: sub_2F3F9C0
// Address: 0x2f3f9c0
//
__int64 __fastcall sub_2F3F9C0(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_4A2AE18;
  v2 = a1[22];
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  return sub_BB9280((__int64)a1);
}
