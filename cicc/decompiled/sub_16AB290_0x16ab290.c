// Function: sub_16AB290
// Address: 0x16ab290
//
__int64 __fastcall sub_16AB290(unsigned int a1, __int64 a2)
{
  unsigned int v2; // ebx
  __int64 *v3; // r15
  unsigned int v4; // ebx
  __int64 result; // rax
  unsigned int v6; // [rsp+8h] [rbp-68h]
  __int64 *v7; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v8; // [rsp+18h] [rbp-58h]
  unsigned __int64 v9; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v10; // [rsp+28h] [rbp-48h]
  __int64 *v11; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v12; // [rsp+38h] [rbp-38h]

  v2 = *(_DWORD *)(a2 + 8);
  v8 = v2;
  if ( v2 > 0x40 )
  {
    sub_16A4FD0((__int64)&v7, (const void **)a2);
    if ( a1 <= v2 )
    {
      v2 = v8;
      goto LABEL_17;
    }
  }
  else
  {
    v7 = *(__int64 **)a2;
    if ( a1 <= v2 )
    {
      v10 = v2;
      goto LABEL_4;
    }
  }
  sub_16A5C50((__int64)&v11, (const void **)a2, a1);
  if ( v8 > 0x40 && v7 )
    j_j___libc_free_0_0(v7);
  v2 = v12;
  v7 = v11;
  v8 = v12;
LABEL_17:
  v10 = v2;
  if ( v2 > 0x40 )
  {
    sub_16A4EF0((__int64)&v9, a1, 0);
    goto LABEL_5;
  }
LABEL_4:
  v9 = a1 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v2);
LABEL_5:
  sub_16AB0A0((__int64)&v11, (__int64)&v7, (__int64)&v9);
  if ( v8 > 0x40 && v7 )
    j_j___libc_free_0_0(v7);
  v3 = v11;
  v4 = v12;
  v7 = v11;
  v8 = v12;
  if ( v10 > 0x40 && v9 )
  {
    j_j___libc_free_0_0(v9);
    v4 = v8;
    v3 = v7;
  }
  if ( v4 > 0x40 )
  {
    if ( v4 - (unsigned int)sub_16A57B0((__int64)&v7) <= 0x40 )
    {
      result = *v3;
      if ( *v3 > (unsigned __int64)a1 )
      {
        j_j___libc_free_0_0(v3);
        return a1;
      }
    }
    else
    {
      result = a1;
    }
    if ( v3 )
    {
      v6 = result;
      j_j___libc_free_0_0(v3);
      return v6;
    }
  }
  else
  {
    result = (unsigned int)v3;
    if ( (unsigned __int64)v3 > a1 )
      return a1;
  }
  return result;
}
