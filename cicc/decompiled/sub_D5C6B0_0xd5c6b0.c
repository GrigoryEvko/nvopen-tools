// Function: sub_D5C6B0
// Address: 0xd5c6b0
//
__int64 __fastcall sub_D5C6B0(unsigned __int8 *a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // edx
  __int64 v6; // rdx
  __int64 v7; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-28h]
  char v9; // [rsp+10h] [rbp-20h]

  if ( *(_BYTE *)a2 == 17 )
  {
    v8 = *(_DWORD *)(a2 + 32);
    if ( v8 > 0x40 )
      sub_C43780((__int64)&v7, (const void **)(a2 + 24));
    else
      v7 = *(_QWORD *)(a2 + 24);
    v9 = 1;
  }
  else
  {
    v5 = *a1;
    if ( (unsigned __int8)(v5 - 2) > 1u )
      return 0;
    sub_D5C200((__int64)&v7, (char *)a2, v5, 0);
    if ( !v9 )
      return 0;
  }
  if ( *(_DWORD *)(a3 + 8) <= 0x40u && v8 <= 0x40 )
  {
    v6 = v7;
    *(_DWORD *)(a3 + 8) = v8;
    *(_QWORD *)a3 = v6;
  }
  else
  {
    sub_C43990(a3, (__int64)&v7);
    if ( v9 )
    {
      v9 = 0;
      if ( v8 > 0x40 )
      {
        if ( v7 )
          j_j___libc_free_0_0(v7);
      }
    }
  }
  return 1;
}
