// Function: sub_3597A30
// Address: 0x3597a30
//
__int64 __fastcall sub_3597A30(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_4A2B598;
  v2 = a1[22];
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  return sub_BB9280((__int64)a1);
}
