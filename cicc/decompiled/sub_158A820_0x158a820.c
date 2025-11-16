// Function: sub_158A820
// Address: 0x158a820
//
__int64 __fastcall sub_158A820(__int64 a1, unsigned __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r13d
  unsigned int v4; // ecx
  unsigned __int64 v5; // rbx
  unsigned int v7; // r12d
  unsigned __int64 *v8; // r14
  unsigned int v9; // r12d
  int v10; // eax
  unsigned __int64 *v11; // rdi
  unsigned __int64 *v12; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v13; // [rsp+8h] [rbp-38h]
  unsigned __int64 *v14; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-28h]

  LOBYTE(v2) = sub_158A0B0(a1);
  v3 = v2;
  if ( !(_BYTE)v2 )
  {
    v13 = *(_DWORD *)(a1 + 24);
    if ( v13 > 0x40 )
      sub_16A4FD0(&v12, a1 + 16);
    else
      v12 = *(unsigned __int64 **)(a1 + 16);
    sub_16A7590(&v12, a1);
    v7 = v13;
    v8 = v12;
    v13 = 0;
    v15 = v7;
    v14 = v12;
    if ( v7 <= 0x40 )
    {
      LOBYTE(v3) = a2 < (unsigned __int64)v12;
      return v3;
    }
    if ( v7 - (unsigned int)sub_16A57B0(&v14) <= 0x40 )
    {
      if ( a2 < *v8 )
      {
        v3 = 1;
LABEL_20:
        j_j___libc_free_0_0(v8);
        if ( v13 <= 0x40 )
          return v3;
        v11 = v12;
        if ( v12 )
          goto LABEL_13;
        return v3;
      }
    }
    else
    {
      v3 = 1;
    }
    if ( !v8 )
      return v3;
    goto LABEL_20;
  }
  v4 = *(_DWORD *)(a1 + 8);
  v5 = a2 - 1;
  v15 = v4;
  if ( v4 <= 0x40 )
  {
    v14 = (unsigned __int64 *)(0xFFFFFFFFFFFFFFFFLL >> -(char)v4);
LABEL_4:
    LOBYTE(v3) = (unsigned __int64)v14 > v5;
    return v3;
  }
  sub_16A4EF0(&v14, -1, 1);
  v9 = v15;
  if ( v15 <= 0x40 )
    goto LABEL_4;
  v10 = sub_16A57B0(&v14);
  v11 = v14;
  if ( v9 - v10 <= 0x40 )
  {
    LOBYTE(v3) = *v14 > v5;
LABEL_13:
    j_j___libc_free_0_0(v11);
    return v3;
  }
  if ( v14 )
    goto LABEL_13;
  return v3;
}
