// Function: sub_25B6400
// Address: 0x25b6400
//
__int64 __fastcall sub_25B6400(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = off_4A1F200;
  v2 = a1[22];
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 56LL))(v2);
  return sub_BB9260((__int64)a1);
}
