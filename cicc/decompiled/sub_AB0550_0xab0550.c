// Function: sub_AB0550
// Address: 0xab0550
//
__int64 __fastcall sub_AB0550(__int64 a1, unsigned __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r12d
  unsigned __int64 v5; // rbx
  unsigned int v6; // r13d
  int v7; // eax
  unsigned __int64 *v8; // rdi
  unsigned int v9; // r13d
  unsigned __int64 *v10; // r14
  unsigned __int64 *v11; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v12; // [rsp+8h] [rbp-38h]
  unsigned __int64 *v13; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-28h]

  LOBYTE(v2) = sub_AAF760(a1);
  v3 = v2;
  if ( !(_BYTE)v2 )
  {
    v12 = *(_DWORD *)(a1 + 24);
    if ( v12 > 0x40 )
      sub_C43780(&v11, a1 + 16);
    else
      v11 = *(unsigned __int64 **)(a1 + 16);
    sub_C46B40(&v11, a1);
    v9 = v12;
    v10 = v11;
    v12 = 0;
    v14 = v9;
    v13 = v11;
    if ( v9 <= 0x40 )
    {
      LOBYTE(v3) = a2 < (unsigned __int64)v11;
      return v3;
    }
    if ( v9 - (unsigned int)sub_C444A0(&v13) <= 0x40 )
    {
      if ( a2 < *v10 )
      {
        v3 = 1;
LABEL_15:
        j_j___libc_free_0_0(v10);
        if ( v12 <= 0x40 )
          return v3;
        v8 = v11;
        goto LABEL_7;
      }
    }
    else
    {
      v3 = 1;
    }
    if ( !v10 )
      return v3;
    goto LABEL_15;
  }
  if ( !a2 )
    return v3;
  v5 = a2 - 1;
  sub_9691E0((__int64)&v13, *(_DWORD *)(a1 + 8), -1, 1u, 0);
  v6 = v14;
  if ( v14 <= 0x40 )
  {
    LOBYTE(v3) = v5 < (unsigned __int64)v13;
    return v3;
  }
  v7 = sub_C444A0(&v13);
  v8 = v13;
  if ( v6 - v7 <= 0x40 )
    LOBYTE(v3) = v5 < *v13;
LABEL_7:
  if ( !v8 )
    return v3;
  j_j___libc_free_0_0(v8);
  return v3;
}
