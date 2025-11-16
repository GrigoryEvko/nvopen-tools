// Function: sub_1CF54D0
// Address: 0x1cf54d0
//
__int64 __fastcall sub_1CF54D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r13
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned int v9; // r13d
  unsigned int v10; // ebx
  __int64 v11; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-48h]
  __int64 v13; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v14; // [rsp+28h] [rbp-38h]

  v2 = 0;
  v3 = *(_QWORD *)(a2 - 48);
  if ( *(_BYTE *)(v3 + 16) != 54 )
    return v2;
  v5 = **(_QWORD **)(a2 - 24);
  if ( *(_BYTE *)(v5 + 8) == 16 )
    v5 = **(_QWORD **)(v5 + 16);
  v12 = 8 * sub_15A95A0(*(_QWORD *)(a1 + 24), *(_DWORD *)(v5 + 8) >> 8);
  if ( v12 > 0x40 )
    sub_16A4EF0((__int64)&v11, 0, 0);
  else
    v11 = 0;
  v6 = sub_164A410(*(_QWORD *)(a2 - 24), *(_QWORD *)(a1 + 24), (__int64)&v11);
  if ( *(_BYTE *)(v6 + 16) != 53 )
    v6 = 0;
  v7 = **(_QWORD **)(v3 - 24);
  if ( *(_BYTE *)(v7 + 8) == 16 )
    v7 = **(_QWORD **)(v7 + 16);
  v14 = 8 * sub_15A95A0(*(_QWORD *)(a1 + 24), *(_DWORD *)(v7 + 8) >> 8);
  if ( v14 > 0x40 )
    sub_16A4EF0((__int64)&v13, 0, 0);
  else
    v13 = 0;
  v8 = sub_164A410(*(_QWORD *)(v3 - 24), *(_QWORD *)(a1 + 24), (__int64)&v13);
  v9 = v12;
  v2 = v8;
  if ( *(_BYTE *)(v8 + 16) != 17 )
    v2 = 0;
  if ( v12 <= 0x40 )
  {
    if ( v11 )
      goto LABEL_30;
  }
  else if ( v9 != (unsigned int)sub_16A57B0((__int64)&v11) )
  {
    goto LABEL_30;
  }
  if ( !v6 || !v2 )
    goto LABEL_30;
  v10 = v14;
  if ( v14 <= 0x40 )
  {
    if ( v13 )
    {
LABEL_24:
      if ( v9 > 0x40 && v11 )
        j_j___libc_free_0_0(v11);
      return 0;
    }
  }
  else if ( (unsigned int)sub_16A57B0((__int64)&v13) != v10 )
  {
    goto LABEL_21;
  }
  if ( !(unsigned __int8)sub_15E0450(v2) )
  {
LABEL_30:
    if ( v14 <= 0x40 )
    {
LABEL_23:
      v9 = v12;
      goto LABEL_24;
    }
LABEL_21:
    if ( v13 )
      j_j___libc_free_0_0(v13);
    goto LABEL_23;
  }
  if ( v14 > 0x40 && v13 )
    j_j___libc_free_0_0(v13);
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0(v11);
  return v2;
}
