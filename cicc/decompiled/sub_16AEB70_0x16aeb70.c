// Function: sub_16AEB70
// Address: 0x16aeb70
//
__int64 __fastcall sub_16AEB70(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  unsigned int v5; // ebx
  _QWORD *v6; // rdx
  unsigned int v7; // eax
  unsigned int v8; // ebx
  const void *v9; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v10; // [rsp+8h] [rbp-48h]
  _QWORD *v11; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-38h]
  const void *v13; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v14; // [rsp+28h] [rbp-28h]

  if ( a4 <= 1 )
  {
    sub_16A9D70(a1, a2, a3);
    return a1;
  }
  v10 = 1;
  v9 = 0;
  v12 = 1;
  v11 = 0;
  sub_16ADD10(a2, a3, (unsigned __int64 *)&v9, (unsigned __int64 *)&v11);
  v5 = v12;
  if ( v12 > 0x40 )
  {
    if ( v5 - (unsigned int)sub_16A57B0((__int64)&v11) > 0x40 )
    {
      v7 = v10;
      goto LABEL_7;
    }
    v6 = (_QWORD *)*v11;
  }
  else
  {
    v6 = v11;
  }
  v7 = v10;
  if ( !v6 )
  {
    *(_DWORD *)(a1 + 8) = v10;
    v10 = 0;
    *(_QWORD *)a1 = v9;
    if ( v5 <= 0x40 )
      goto LABEL_10;
    goto LABEL_14;
  }
LABEL_7:
  v14 = v7;
  if ( v7 > 0x40 )
    sub_16A4FD0((__int64)&v13, &v9);
  else
    v13 = v9;
  sub_16A7490((__int64)&v13, 1);
  v8 = v12;
  *(_DWORD *)(a1 + 8) = v14;
  *(_QWORD *)a1 = v13;
  if ( v8 > 0x40 )
  {
LABEL_14:
    if ( v11 )
      j_j___libc_free_0_0(v11);
  }
LABEL_10:
  if ( v10 <= 0x40 || !v9 )
    return a1;
  j_j___libc_free_0_0(v9);
  return a1;
}
