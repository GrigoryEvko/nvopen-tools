// Function: sub_16A7C10
// Address: 0x16a7c10
//
__int64 __fastcall sub_16A7C10(__int64 a1, __int64 *a2)
{
  __int64 v3; // [rsp+0h] [rbp-20h] BYREF
  int v4; // [rsp+8h] [rbp-18h]

  sub_16A7B50((__int64)&v3, a1, a2);
  if ( *(_DWORD *)(a1 + 8) > 0x40u && *(_QWORD *)a1 )
    j_j___libc_free_0_0(*(_QWORD *)a1);
  *(_QWORD *)a1 = v3;
  *(_DWORD *)(a1 + 8) = v4;
  return a1;
}
