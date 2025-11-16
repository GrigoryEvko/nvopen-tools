// Function: sub_D87290
// Address: 0xd87290
//
__int64 __fastcall sub_D87290(__int64 a1, __int64 a2, __int64 a3)
{
  bool v5; // cc
  unsigned int v6; // eax
  __int64 v7; // rdi
  __int64 v8; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v9; // [rsp+8h] [rbp-28h]
  __int64 v10; // [rsp+10h] [rbp-20h]
  int v11; // [rsp+18h] [rbp-18h]

  sub_AB3510(a1, a2, a3, 0);
  if ( !sub_AB0120(a1) )
    return a1;
  sub_AADB10((__int64)&v8, *(_DWORD *)(a1 + 8), 1);
  if ( *(_DWORD *)(a1 + 8) > 0x40u && *(_QWORD *)a1 )
    j_j___libc_free_0_0(*(_QWORD *)a1);
  v5 = *(_DWORD *)(a1 + 24) <= 0x40u;
  *(_QWORD *)a1 = v8;
  v6 = v9;
  v9 = 0;
  *(_DWORD *)(a1 + 8) = v6;
  if ( !v5 )
  {
    v7 = *(_QWORD *)(a1 + 16);
    if ( v7 )
    {
      j_j___libc_free_0_0(v7);
      v5 = v9 <= 0x40;
      *(_QWORD *)(a1 + 16) = v10;
      *(_DWORD *)(a1 + 24) = v11;
      if ( !v5 )
      {
        if ( v8 )
          j_j___libc_free_0_0(v8);
      }
      return a1;
    }
  }
  *(_QWORD *)(a1 + 16) = v10;
  *(_DWORD *)(a1 + 24) = v11;
  return a1;
}
