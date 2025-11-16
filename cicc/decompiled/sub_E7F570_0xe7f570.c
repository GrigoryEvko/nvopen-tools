// Function: sub_E7F570
// Address: 0xe7f570
//
__int64 __fastcall sub_E7F570(__int64 a1, unsigned __int8 *a2, unsigned int a3, __int64 a4)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9

  if ( sub_E7E4B0(a1) )
    sub_C64ED0("Emitting values inside a locked bundle is forbidden", 1u);
  sub_E7E6A0(a1, a2, v6, v7, v8, v9);
  return sub_E8D730(a1, a2, a3, a4);
}
