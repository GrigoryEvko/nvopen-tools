// Function: sub_1F6F9E0
// Address: 0x1f6f9e0
//
__int64 __fastcall sub_1F6F9E0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int16 v3; // r14
  unsigned int v4; // ebx
  const void **v5; // r13
  unsigned __int64 *v6; // r12
  unsigned __int64 v8; // rax
  unsigned int v9; // eax
  unsigned __int64 v10; // r13
  unsigned __int64 v11; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v12; // [rsp+8h] [rbp-38h]
  unsigned __int64 v13; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-28h]

  v2 = *(_QWORD *)(*(_QWORD *)a2 + 88LL);
  v3 = *(_WORD *)(*(_QWORD *)a2 + 26LL);
  v4 = *(_DWORD *)(v2 + 32);
  if ( v4 <= 0x40 )
  {
    LODWORD(v6) = 0;
    if ( !*(_QWORD *)(v2 + 24) )
      return (unsigned int)v6;
    if ( (v3 & 8) != 0 )
      return (unsigned int)v6;
    v8 = *(_QWORD *)(v2 + 24);
    if ( v8 )
    {
      LODWORD(v6) = 1;
      if ( (v8 & (v8 - 1)) == 0 )
        return (unsigned int)v6;
    }
    v12 = v4;
    v6 = &v11;
    goto LABEL_10;
  }
  v5 = (const void **)(v2 + 24);
  LODWORD(v6) = 0;
  if ( v4 == (unsigned int)sub_16A57B0(v2 + 24) )
    return (unsigned int)v6;
  if ( (v3 & 8) != 0 )
    return (unsigned int)v6;
  LODWORD(v6) = 1;
  if ( (unsigned int)sub_16A5940((__int64)v5) == 1 )
    return (unsigned int)v6;
  v6 = &v11;
  v12 = v4;
  sub_16A4FD0((__int64)&v11, v5);
  LOBYTE(v4) = v12;
  if ( v12 <= 0x40 )
  {
    v8 = v11;
LABEL_10:
    v11 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v4) & ~v8;
    goto LABEL_11;
  }
  sub_16A8F40((__int64 *)&v11);
LABEL_11:
  sub_16A7400((__int64)&v11);
  v9 = v12;
  v10 = v11;
  v12 = 0;
  v14 = v9;
  v13 = v11;
  if ( v9 > 0x40 )
  {
    LOBYTE(v6) = (unsigned int)sub_16A5940((__int64)&v13) == 1;
    if ( v10 )
    {
      j_j___libc_free_0_0(v10);
      if ( v12 > 0x40 )
      {
        if ( v11 )
          j_j___libc_free_0_0(v11);
      }
    }
  }
  else
  {
    LODWORD(v6) = 0;
    if ( v11 )
      LOBYTE(v6) = (v11 & (v11 - 1)) == 0;
  }
  return (unsigned int)v6;
}
