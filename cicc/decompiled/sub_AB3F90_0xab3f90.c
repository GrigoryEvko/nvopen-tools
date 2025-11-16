// Function: sub_AB3F90
// Address: 0xab3f90
//
__int64 __fastcall sub_AB3F90(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v5; // ebx
  char v6; // al
  __int64 v7; // r8
  bool v8; // al
  unsigned int v9; // r15d
  __int64 v10; // r12
  unsigned int v11; // eax
  unsigned int v12; // eax
  __int64 v13; // [rsp+8h] [rbp-68h]
  __int64 v14; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-58h]
  __int64 v16; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v17; // [rsp+28h] [rbp-48h]
  __int64 v18; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v19; // [rsp+38h] [rbp-38h]

  if ( sub_AAF7D0(a2) )
  {
    sub_AADB10(a1, a3, 0);
    return a1;
  }
  v5 = *(_DWORD *)(a2 + 8);
  v6 = sub_AAF760(a2);
  v7 = a2 + 16;
  if ( !v6 )
  {
    v8 = sub_AB0100(a2);
    v7 = a2 + 16;
    if ( !v8 )
    {
      sub_C449B0(&v18, a2 + 16, a3);
      sub_C449B0(&v16, a2, a3);
      sub_AADC30(a1, (__int64)&v16, &v18);
      sub_969240(&v16);
      sub_969240(&v18);
      return a1;
    }
  }
  v15 = a3;
  if ( a3 > 0x40 )
  {
    v13 = v7;
    sub_C43690(&v14, 0, 0);
    v7 = v13;
  }
  else
  {
    v14 = 0;
  }
  v9 = *(_DWORD *)(a2 + 24);
  if ( v9 <= 0x40 )
  {
    if ( *(_QWORD *)(a2 + 16) )
      goto LABEL_10;
  }
  else if ( v9 != (unsigned int)sub_C444A0(v7) )
  {
    goto LABEL_10;
  }
  sub_C449B0(&v18, a2, a3);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  v14 = v18;
  v12 = v19;
  v19 = 0;
  v15 = v12;
  sub_969240(&v18);
LABEL_10:
  v17 = a3;
  v10 = 1LL << v5;
  if ( a3 <= 0x40 )
  {
    v16 = 0;
LABEL_12:
    v16 |= v10;
    goto LABEL_13;
  }
  sub_C43690(&v16, 0, 0);
  if ( v17 <= 0x40 )
    goto LABEL_12;
  *(_QWORD *)(v16 + 8LL * (v5 >> 6)) |= v10;
LABEL_13:
  v11 = v15;
  v15 = 0;
  v19 = v11;
  v18 = v14;
  sub_AADC30(a1, (__int64)&v18, &v16);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  return a1;
}
