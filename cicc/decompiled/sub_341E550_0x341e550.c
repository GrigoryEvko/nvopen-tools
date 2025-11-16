// Function: sub_341E550
// Address: 0x341e550
//
__int64 __fastcall sub_341E550(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_4A36A08;
  v2 = a1[25];
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
