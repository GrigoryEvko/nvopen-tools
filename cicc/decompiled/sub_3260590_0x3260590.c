// Function: sub_3260590
// Address: 0x3260590
//
__int64 __fastcall sub_3260590(__int64 a1, __int64 a2, int a3)
{
  unsigned int v3; // r13d
  unsigned int v4; // r13d
  __int64 result; // rax
  __int64 v6; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v7; // [rsp+8h] [rbp-28h]

  v3 = *(_DWORD *)(a2 + 8);
  if ( *(_DWORD *)(a1 + 8) >= v3 )
    v3 = *(_DWORD *)(a1 + 8);
  v4 = a3 + v3;
  sub_C449B0((__int64)&v6, (const void **)a1, v4);
  if ( *(_DWORD *)(a1 + 8) > 0x40u && *(_QWORD *)a1 )
    j_j___libc_free_0_0(*(_QWORD *)a1);
  *(_QWORD *)a1 = v6;
  *(_DWORD *)(a1 + 8) = v7;
  sub_C449B0((__int64)&v6, (const void **)a2, v4);
  if ( *(_DWORD *)(a2 + 8) > 0x40u && *(_QWORD *)a2 )
    j_j___libc_free_0_0(*(_QWORD *)a2);
  *(_QWORD *)a2 = v6;
  result = v7;
  *(_DWORD *)(a2 + 8) = v7;
  return result;
}
