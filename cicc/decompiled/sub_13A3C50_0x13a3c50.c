// Function: sub_13A3C50
// Address: 0x13a3c50
//
__int64 __fastcall sub_13A3C50(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // eax
  _QWORD *v5; // rdx
  unsigned int v6; // ebx
  unsigned int v7; // eax
  _QWORD *v9; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v10; // [rsp+18h] [rbp-58h]
  _QWORD *v11; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v12; // [rsp+28h] [rbp-48h]
  _QWORD *v13; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v14; // [rsp+38h] [rbp-38h]

  v4 = *(_DWORD *)(a2 + 8);
  v10 = v4;
  if ( v4 <= 0x40 )
  {
    v5 = *(_QWORD **)a2;
    v12 = v4;
    v9 = v5;
LABEL_3:
    v11 = *(_QWORD **)a2;
    goto LABEL_4;
  }
  sub_16A4FD0(&v9, a2);
  v12 = *(_DWORD *)(a2 + 8);
  if ( v12 <= 0x40 )
    goto LABEL_3;
  sub_16A4FD0(&v11, a2);
LABEL_4:
  sub_16AE5C0(a2, a3, &v9, &v11);
  v6 = v12;
  if ( v12 > 0x40 )
  {
    if ( v6 - (unsigned int)sub_16A57B0(&v11) <= 0x40 && !*v11 )
      goto LABEL_9;
  }
  else if ( !v11 )
  {
LABEL_9:
    v7 = v10;
    v10 = 0;
    *(_DWORD *)(a1 + 8) = v7;
    *(_QWORD *)a1 = v9;
    goto LABEL_10;
  }
  if ( sub_13A39D0(a2, 0) )
  {
    if ( sub_13A39D0(a3, 0) )
      goto LABEL_9;
    if ( !sub_13A3940(a2, 0) )
      goto LABEL_22;
  }
  else if ( !sub_13A3940(a2, 0) )
  {
    goto LABEL_22;
  }
  if ( sub_13A3940(a3, 0) )
    goto LABEL_9;
LABEL_22:
  v14 = v10;
  if ( v10 > 0x40 )
    sub_16A4FD0(&v13, &v9);
  else
    v13 = v9;
  sub_16A7800(&v13, 1);
  v6 = v12;
  *(_DWORD *)(a1 + 8) = v14;
  *(_QWORD *)a1 = v13;
LABEL_10:
  if ( v6 > 0x40 && v11 )
    j_j___libc_free_0_0(v11);
  if ( v10 > 0x40 && v9 )
    j_j___libc_free_0_0(v9);
  return a1;
}
