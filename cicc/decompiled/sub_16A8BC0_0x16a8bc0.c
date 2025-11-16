// Function: sub_16A8BC0
// Address: 0x16a8bc0
//
__int64 __fastcall sub_16A8BC0(__int64 a1, unsigned int a2, __int64 a3)
{
  unsigned int i; // r12d
  unsigned int v5; // edx
  unsigned __int64 v6; // rax
  unsigned int v7; // ecx
  unsigned __int64 v9; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v10; // [rsp+8h] [rbp-28h]

  sub_16A5DD0(a1, a3, a2);
  for ( i = *(_DWORD *)(a3 + 8); a2 > i; i *= 2 )
  {
    v7 = *(_DWORD *)(a1 + 8);
    v10 = v7;
    if ( v7 <= 0x40 )
    {
      v5 = v7;
      v9 = *(_QWORD *)a1;
    }
    else
    {
      sub_16A4FD0((__int64)&v9, (const void **)a1);
      v7 = v10;
      if ( v10 > 0x40 )
      {
        sub_16A7DC0((__int64 *)&v9, i);
        if ( *(_DWORD *)(a1 + 8) <= 0x40u )
          goto LABEL_7;
        goto LABEL_15;
      }
      v5 = *(_DWORD *)(a1 + 8);
    }
    v6 = 0;
    if ( i != v7 )
      v6 = (v9 << i) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v7);
    v9 = v6;
    if ( v5 <= 0x40 )
    {
LABEL_7:
      *(_QWORD *)a1 |= v9;
      goto LABEL_8;
    }
LABEL_15:
    sub_16A89F0((__int64 *)a1, (__int64 *)&v9);
LABEL_8:
    if ( v10 > 0x40 && v9 )
      j_j___libc_free_0_0(v9);
  }
  return a1;
}
