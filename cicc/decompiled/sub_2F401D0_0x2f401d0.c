// Function: sub_2F401D0
// Address: 0x2f401d0
//
__int64 __fastcall sub_2F401D0(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_4A2AE18;
  v2 = a1[22];
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  return sub_BB9280((__int64)a1);
}
