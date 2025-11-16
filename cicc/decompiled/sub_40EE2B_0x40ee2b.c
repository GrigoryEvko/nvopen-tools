// Function: sub_40EE2B
// Address: 0x40ee2b
//
__int64 __fastcall sub_40EE2B(_DWORD *a1, __int64 *a2, const char **a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v8; // r8
  __int64 v9; // r8
  __int64 v10; // r8
  __int64 v11; // r8
  __int64 v12; // r8
  __int64 v13; // r8

  if ( a2 && *a1 == 2 )
    sub_40ECF5((int)a1, a2, (__int64)a3, a4, a5, a6, a5);
  sub_40EDDD((__int64)a1, (__int64)"num_ops", 5, a3 + 2, a5);
  sub_40EDDD((__int64)a1, (__int64)"num_wait", 5, a3 + 7, v8);
  sub_40EDDD((__int64)a1, (__int64)"num_spin_acq", 5, a3 + 12, v9);
  sub_40EDDD((__int64)a1, (__int64)"num_owner_switch", 5, a3 + 17, v10);
  sub_40EDDD((__int64)a1, (__int64)"total_wait_time", 5, a3 + 22, v11);
  sub_40EDDD((__int64)a1, (__int64)"max_wait_time", 5, a3 + 27, v12);
  return sub_40EDDD((__int64)a1, (__int64)"max_num_thds", 4, (const char **)(a4 + 16), v13);
}
