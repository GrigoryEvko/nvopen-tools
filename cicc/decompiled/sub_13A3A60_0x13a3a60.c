// Function: sub_13A3A60
// Address: 0x13a3a60
//
__int64 __fastcall sub_13A3A60(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // eax
  _QWORD *v5; // rdx
  unsigned int v6; // ebx
  unsigned int v7; // ebx
  unsigned int v9; // eax
  _QWORD *v10; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v11; // [rsp+18h] [rbp-58h]
  _QWORD *v12; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v13; // [rsp+28h] [rbp-48h]
  _QWORD *v14; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v15; // [rsp+38h] [rbp-38h]

  v4 = *(_DWORD *)(a2 + 8);
  v11 = v4;
  if ( v4 <= 0x40 )
  {
    v5 = *(_QWORD **)a2;
    v13 = v4;
    v10 = v5;
LABEL_3:
    v12 = *(_QWORD **)a2;
    goto LABEL_4;
  }
  sub_16A4FD0(&v10, a2);
  v13 = *(_DWORD *)(a2 + 8);
  if ( v13 <= 0x40 )
    goto LABEL_3;
  sub_16A4FD0(&v12, a2);
LABEL_4:
  sub_16AE5C0(a2, a3, &v10, &v12);
  v6 = v13;
  if ( v13 > 0x40 )
  {
    if ( v6 - (unsigned int)sub_16A57B0(&v12) <= 0x40 && !*v12 )
      goto LABEL_16;
  }
  else if ( !v12 )
  {
    goto LABEL_16;
  }
  if ( (!sub_13A39D0(a2, 0) || !sub_13A39D0(a3, 0)) && (!sub_13A3940(a2, 0) || !sub_13A3940(a3, 0)) )
  {
LABEL_16:
    v9 = v11;
    v11 = 0;
    *(_DWORD *)(a1 + 8) = v9;
    *(_QWORD *)a1 = v10;
    if ( v6 <= 0x40 )
      goto LABEL_11;
    goto LABEL_17;
  }
  v15 = v11;
  if ( v11 > 0x40 )
    sub_16A4FD0(&v14, &v10);
  else
    v14 = v10;
  sub_16A7490(&v14, 1);
  v7 = v13;
  *(_DWORD *)(a1 + 8) = v15;
  *(_QWORD *)a1 = v14;
  if ( v7 <= 0x40 )
    goto LABEL_11;
LABEL_17:
  if ( v12 )
    j_j___libc_free_0_0(v12);
LABEL_11:
  if ( v11 > 0x40 && v10 )
    j_j___libc_free_0_0(v10);
  return a1;
}
