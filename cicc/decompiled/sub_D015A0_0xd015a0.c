// Function: sub_D015A0
// Address: 0xd015a0
//
__int64 __fastcall sub_D015A0(__int64 a1)
{
  unsigned int v1; // r12d
  __int64 v3; // rax
  __int64 v4; // rdx
  int v5; // r13d
  __int64 v6; // rax
  __int64 v7; // rdx
  int v8; // eax
  unsigned __int64 v9; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v10; // [rsp+8h] [rbp-38h]
  __int64 v11; // [rsp+10h] [rbp-30h] BYREF
  __int64 v12; // [rsp+18h] [rbp-28h]

  v1 = *(unsigned __int8 *)(a1 + 48);
  if ( !(_BYTE)v1 )
  {
    v3 = sub_BCAE30(*(_QWORD *)(*(_QWORD *)a1 + 8LL));
    v12 = v4;
    v11 = v3;
    v5 = sub_CA1930(&v11);
    v6 = sub_BCAE30(*(_QWORD *)(*(_QWORD *)a1 + 8LL));
    v12 = v7;
    v11 = v6;
    v8 = sub_CA1930(&v11) + *(_DWORD *)(a1 + 8) + *(_DWORD *)(a1 + 12) - *(_DWORD *)(a1 + 16);
    if ( v8 - v5 > 0 )
    {
      v10 = v8 - v5;
      v1 = a1 + 24;
      if ( (unsigned int)(v8 - v5) > 0x40 )
        sub_C43690((__int64)&v9, -1, 1);
      else
        v9 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v5 + 64 - v8);
      sub_C449B0((__int64)&v11, (const void **)&v9, *(_DWORD *)(a1 + 32));
      LOBYTE(v1) = (int)sub_C49970(a1 + 24, (unsigned __int64 *)&v11) <= 0;
      if ( (unsigned int)v12 > 0x40 && v11 )
        j_j___libc_free_0_0(v11);
      if ( v10 > 0x40 && v9 )
        j_j___libc_free_0_0(v9);
    }
  }
  return v1;
}
