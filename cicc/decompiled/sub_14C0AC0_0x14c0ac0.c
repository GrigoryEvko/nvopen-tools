// Function: sub_14C0AC0
// Address: 0x14c0ac0
//
char __fastcall sub_14C0AC0(
        unsigned __int8 a1,
        __int64 *a2,
        __int64 *a3,
        unsigned __int8 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        __int64 *a8)
{
  __int64 v10; // r15
  bool v11; // cc
  unsigned int v12; // eax
  __int64 v13; // rdi
  char result; // al
  int v15; // [rsp+4h] [rbp-7Ch]
  int v16; // [rsp+4h] [rbp-7Ch]
  __int64 v18; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-68h]
  __int64 v20; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v21; // [rsp+28h] [rbp-58h]
  __int64 v22; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v23; // [rsp+38h] [rbp-48h]
  __int64 v24; // [rsp+40h] [rbp-40h]
  int v25; // [rsp+48h] [rbp-38h]

  v10 = a5 + 16;
  sub_14B86A0(a3, a5, a7 + 1, a8);
  if ( *(_DWORD *)(a5 + 8) <= 0x40u )
  {
    if ( *(_QWORD *)a5 )
      goto LABEL_3;
  }
  else
  {
    v15 = *(_DWORD *)(a5 + 8);
    if ( v15 != (unsigned int)sub_16A57B0(a5) )
      goto LABEL_3;
  }
  if ( *(_DWORD *)(a5 + 24) <= 0x40u )
  {
    result = *(_QWORD *)(a5 + 16) == 0;
  }
  else
  {
    v16 = *(_DWORD *)(a5 + 24);
    result = v16 == (unsigned int)sub_16A57B0(v10);
  }
  if ( !result || a4 )
  {
LABEL_3:
    sub_14B86A0(a2, a6, a7 + 1, a8);
    v19 = *(_DWORD *)(a5 + 8);
    if ( v19 > 0x40 )
      sub_16A4FD0(&v18, a5);
    else
      v18 = *(_QWORD *)a5;
    v21 = *(_DWORD *)(a5 + 24);
    if ( v21 > 0x40 )
      sub_16A4FD0(&v20, v10);
    else
      v20 = *(_QWORD *)(a5 + 16);
    sub_16C05A0(&v22, a1, a4, a6, &v18);
    if ( *(_DWORD *)(a5 + 8) > 0x40u && *(_QWORD *)a5 )
      j_j___libc_free_0_0(*(_QWORD *)a5);
    v11 = *(_DWORD *)(a5 + 24) <= 0x40u;
    *(_QWORD *)a5 = v22;
    v12 = v23;
    v23 = 0;
    *(_DWORD *)(a5 + 8) = v12;
    if ( v11 || (v13 = *(_QWORD *)(a5 + 16)) == 0 )
    {
      *(_QWORD *)(a5 + 16) = v24;
      result = v25;
      *(_DWORD *)(a5 + 24) = v25;
    }
    else
    {
      j_j___libc_free_0_0(v13);
      v11 = v23 <= 0x40;
      *(_QWORD *)(a5 + 16) = v24;
      result = v25;
      *(_DWORD *)(a5 + 24) = v25;
      if ( !v11 && v22 )
      {
        result = j_j___libc_free_0_0(v22);
        if ( v21 <= 0x40 )
          goto LABEL_17;
        goto LABEL_15;
      }
    }
    if ( v21 <= 0x40 )
    {
      if ( v19 <= 0x40 )
        return result;
      goto LABEL_18;
    }
LABEL_15:
    if ( v20 )
      result = j_j___libc_free_0_0(v20);
LABEL_17:
    if ( v19 <= 0x40 )
      return result;
LABEL_18:
    if ( v18 )
      return j_j___libc_free_0_0(v18);
  }
  return result;
}
