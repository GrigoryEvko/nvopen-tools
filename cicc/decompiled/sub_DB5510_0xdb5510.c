// Function: sub_DB5510
// Address: 0xdb5510
//
__int64 __fastcall sub_DB5510(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // ebx
  _QWORD *v4; // rax
  unsigned int v6; // eax
  _QWORD *v7; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-28h]

  sub_DB4FC0((__int64)&v7, a2, a3);
  v3 = v8;
  if ( v8 > 0x40 )
  {
    if ( v3 - (unsigned int)sub_C444A0((__int64)&v7) > 0x40 || *v7 )
    {
      *(_DWORD *)(a1 + 8) = v3;
      sub_C43780(a1, (const void **)&v7);
      v6 = v8;
    }
    else
    {
      *(_DWORD *)(a1 + 8) = v3;
      sub_C43690(a1, 1, 0);
      v6 = v8;
    }
    if ( v6 <= 0x40 || !v7 )
      return a1;
    j_j___libc_free_0_0(v7);
    return a1;
  }
  else
  {
    v4 = v7;
    if ( v7 )
    {
      *(_DWORD *)(a1 + 8) = v8;
      *(_QWORD *)a1 = v4;
      return a1;
    }
    *(_DWORD *)(a1 + 8) = v8;
    *(_QWORD *)a1 = 1;
    return a1;
  }
}
