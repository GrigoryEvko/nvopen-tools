// Function: sub_DB56A0
// Address: 0xdb56a0
//
__int64 __fastcall sub_DB56A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v5; // r14d
  unsigned int v6; // eax
  unsigned int v7; // eax
  const void *v9; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v10; // [rsp+8h] [rbp-28h]

  v5 = *(_DWORD *)(a3 + 8);
  v6 = sub_DB55F0(a2, a4);
  if ( v6 )
  {
    if ( v5 > v6 )
    {
      sub_C44740((__int64)&v9, (char **)a3, v6);
      sub_C449B0(a1, &v9, v5);
      if ( v10 > 0x40 && v9 )
        j_j___libc_free_0_0(v9);
    }
    else
    {
      v7 = *(_DWORD *)(a3 + 8);
      *(_DWORD *)(a1 + 8) = v7;
      if ( v7 > 0x40 )
        sub_C43780(a1, (const void **)a3);
      else
        *(_QWORD *)a1 = *(_QWORD *)a3;
    }
  }
  else
  {
    *(_DWORD *)(a1 + 8) = v5;
    if ( v5 > 0x40 )
      sub_C43690(a1, 0, 0);
    else
      *(_QWORD *)a1 = 0;
  }
  return a1;
}
