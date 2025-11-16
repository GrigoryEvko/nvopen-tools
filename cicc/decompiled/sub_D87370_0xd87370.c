// Function: sub_D87370
// Address: 0xd87370
//
__int64 __fastcall sub_D87370(__int64 a1, __int64 a2)
{
  bool v3; // cc
  unsigned int v4; // eax
  __int64 v5; // rdi
  __int64 result; // rax
  __int64 v7; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-28h]
  __int64 v9; // [rsp+10h] [rbp-20h]
  unsigned int v10; // [rsp+18h] [rbp-18h]

  sub_D87290((__int64)&v7, a1, a2);
  if ( *(_DWORD *)(a1 + 8) > 0x40u && *(_QWORD *)a1 )
    j_j___libc_free_0_0(*(_QWORD *)a1);
  v3 = *(_DWORD *)(a1 + 24) <= 0x40u;
  *(_QWORD *)a1 = v7;
  v4 = v8;
  v8 = 0;
  *(_DWORD *)(a1 + 8) = v4;
  if ( v3 || (v5 = *(_QWORD *)(a1 + 16)) == 0 )
  {
    *(_QWORD *)(a1 + 16) = v9;
    result = v10;
    *(_DWORD *)(a1 + 24) = v10;
  }
  else
  {
    j_j___libc_free_0_0(v5);
    v3 = v8 <= 0x40;
    *(_QWORD *)(a1 + 16) = v9;
    result = v10;
    *(_DWORD *)(a1 + 24) = v10;
    if ( !v3 )
    {
      if ( v7 )
        return j_j___libc_free_0_0(v7);
    }
  }
  return result;
}
