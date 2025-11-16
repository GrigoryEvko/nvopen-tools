// Function: sub_16AA580
// Address: 0x16aa580
//
__int64 __fastcall sub_16AA580(__int64 a1, __int64 a2, __int64 a3, bool *a4)
{
  unsigned int v7; // ebx
  unsigned int v8; // ebx
  unsigned int v10; // ebx
  bool v11; // al
  __int64 v12; // rdi
  __int64 v13; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-48h]
  __int64 v15; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v16; // [rsp+28h] [rbp-38h]

  sub_16A7B50(a1, a2, (__int64 *)a3);
  v7 = *(_DWORD *)(a2 + 8);
  if ( v7 > 0x40 )
  {
    if ( v7 - (unsigned int)sub_16A57B0(a2) <= 0x40 && !**(_QWORD **)a2 )
      goto LABEL_5;
  }
  else if ( !*(_QWORD *)a2 )
  {
LABEL_5:
    *a4 = 0;
    return a1;
  }
  v8 = *(_DWORD *)(a3 + 8);
  if ( v8 > 0x40 )
  {
    if ( v8 - (unsigned int)sub_16A57B0(a3) <= 0x40 && !**(_QWORD **)a3 )
      goto LABEL_5;
  }
  else if ( !*(_QWORD *)a3 )
  {
    goto LABEL_5;
  }
  sub_16A9D70((__int64)&v13, a1, a3);
  v10 = v14;
  if ( v14 <= 0x40 )
  {
    if ( v13 != *(_QWORD *)a2 )
      goto LABEL_14;
LABEL_19:
    sub_16A9D70((__int64)&v15, a1, a2);
    if ( v16 <= 0x40 )
    {
      *a4 = v15 != *(_QWORD *)a3;
    }
    else
    {
      v11 = sub_16A5220((__int64)&v15, (const void **)a3);
      v12 = v15;
      *a4 = !v11;
      if ( v12 )
        j_j___libc_free_0_0(v12);
    }
    v10 = v14;
    goto LABEL_15;
  }
  if ( sub_16A5220((__int64)&v13, (const void **)a2) )
    goto LABEL_19;
LABEL_14:
  *a4 = 1;
LABEL_15:
  if ( v10 > 0x40 && v13 )
    j_j___libc_free_0_0(v13);
  return a1;
}
