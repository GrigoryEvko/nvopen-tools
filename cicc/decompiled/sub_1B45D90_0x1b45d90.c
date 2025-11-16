// Function: sub_1B45D90
// Address: 0x1b45d90
//
__int64 __fastcall sub_1B45D90(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  unsigned __int64 v6; // rax
  __int64 *v7; // rax

  v6 = sub_157EBA0(a3);
  v7 = sub_1B44C50(a1, v6);
  if ( v7 && v7 == sub_1B44C50(a1, a2) )
    return sub_1B45440(a1, a2, a3, a4);
  else
    return 0;
}
