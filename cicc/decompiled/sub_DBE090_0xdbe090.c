// Function: sub_DBE090
// Address: 0xdbe090
//
__int64 __fastcall sub_DBE090(__int64 a1, __int64 a2)
{
  _QWORD **v2; // r12
  __int64 v3; // rax
  unsigned int v4; // ebx
  _QWORD *v6; // [rsp+0h] [rbp-20h] BYREF
  unsigned int v7; // [rsp+8h] [rbp-18h]

  for ( ; *(_WORD *)(a2 + 24) == 4; a2 = *(_QWORD *)(a2 + 32) )
    ;
  v2 = &v6;
  v3 = sub_DBB9F0(a1, a2, 0, 0);
  sub_AB0A00((__int64)&v6, v3);
  v4 = v7;
  if ( v7 <= 0x40 )
  {
    LOBYTE(v2) = v6 != 0;
    return (unsigned int)v2;
  }
  else
  {
    if ( v4 - (unsigned int)sub_C444A0((__int64)&v6) <= 0x40 )
    {
      LOBYTE(v2) = *v6 != 0;
    }
    else
    {
      LODWORD(v2) = 1;
      if ( !v6 )
        return 1;
    }
    j_j___libc_free_0_0(v6);
    return (unsigned int)v2;
  }
}
