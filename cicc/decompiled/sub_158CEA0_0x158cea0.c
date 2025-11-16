// Function: sub_158CEA0
// Address: 0x158cea0
//
__int64 __fastcall sub_158CEA0(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v6; // ebx
  char v7; // al
  __int64 v8; // r8
  bool v9; // al
  unsigned int v10; // r15d
  __int64 v11; // r12
  unsigned int v12; // eax
  __int64 v13; // rdi
  __int64 v14; // [rsp+8h] [rbp-68h]
  __int64 v15; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v16; // [rsp+18h] [rbp-58h]
  __int64 v17; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v18; // [rsp+28h] [rbp-48h]
  __int64 v19; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v20; // [rsp+38h] [rbp-38h]

  if ( sub_158A120(a2) )
  {
    sub_15897D0(a1, a3, 0);
    return a1;
  }
  v6 = *(_DWORD *)(a2 + 8);
  v7 = sub_158A0B0(a2);
  v8 = a2 + 16;
  if ( !v7 )
  {
    v9 = sub_158A670(a2);
    v8 = a2 + 16;
    if ( !v9 )
    {
      sub_16A5C50(&v19, a2 + 16, a3);
      sub_16A5C50(&v17, a2, a3);
      sub_15898E0(a1, (__int64)&v17, &v19);
      if ( v18 > 0x40 && v17 )
        j_j___libc_free_0_0(v17);
      if ( v20 > 0x40 )
      {
        v13 = v19;
        if ( v19 )
          goto LABEL_20;
      }
      return a1;
    }
  }
  v16 = a3;
  if ( a3 <= 0x40 )
  {
    v10 = *(_DWORD *)(a2 + 24);
    v15 = 0;
    if ( v10 > 0x40 )
      goto LABEL_8;
LABEL_24:
    if ( *(_QWORD *)(a2 + 16) )
      goto LABEL_9;
    goto LABEL_25;
  }
  v14 = v8;
  sub_16A4EF0(&v15, 0, 0);
  v10 = *(_DWORD *)(a2 + 24);
  v8 = v14;
  if ( v10 <= 0x40 )
    goto LABEL_24;
LABEL_8:
  if ( v10 != (unsigned int)sub_16A57B0(v8) )
    goto LABEL_9;
LABEL_25:
  sub_16A5C50(&v19, a2, a3);
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  v15 = v19;
  v16 = v20;
LABEL_9:
  v18 = a3;
  v11 = 1LL << v6;
  if ( a3 > 0x40 )
  {
    sub_16A4EF0(&v17, 0, 0);
    if ( v18 > 0x40 )
    {
      *(_QWORD *)(v17 + 8LL * (v6 >> 6)) |= v11;
      goto LABEL_12;
    }
  }
  else
  {
    v17 = 0;
  }
  v17 |= v11;
LABEL_12:
  v12 = v16;
  v16 = 0;
  v20 = v12;
  v19 = v15;
  sub_15898E0(a1, (__int64)&v19, &v17);
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  if ( v16 > 0x40 )
  {
    v13 = v15;
    if ( v15 )
LABEL_20:
      j_j___libc_free_0_0(v13);
  }
  return a1;
}
