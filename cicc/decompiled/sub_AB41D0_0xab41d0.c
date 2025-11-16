// Function: sub_AB41D0
// Address: 0xab41d0
//
__int64 __fastcall sub_AB41D0(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v6; // eax
  __int64 v7; // rsi
  unsigned int v8; // ebx
  int v9; // ebx
  unsigned int v10; // r15d
  unsigned __int64 v11; // rax
  unsigned int v12; // eax
  __int64 v13; // rsi
  __int64 v14; // rdi
  __int64 v16; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v17; // [rsp+8h] [rbp-58h]
  __int64 v18; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-48h]
  __int64 v20; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v21; // [rsp+28h] [rbp-38h]

  if ( sub_AAF7D0(a2) )
  {
    sub_AADB10(a1, a3, 0);
    return a1;
  }
  v6 = *(_DWORD *)(a2 + 24);
  v7 = *(_QWORD *)(a2 + 16);
  v8 = v6 - 1;
  if ( v6 <= 0x40 )
  {
    if ( v7 != 1LL << v8 )
      goto LABEL_5;
LABEL_24:
    sub_C449B0(&v20, a2 + 16, a3);
    sub_C44830(&v18, a2, a3);
    sub_AADC30(a1, (__int64)&v18, &v20);
    if ( v19 > 0x40 && v18 )
      j_j___libc_free_0_0(v18);
    if ( v21 <= 0x40 )
      return a1;
    v14 = v20;
    if ( !v20 )
      return a1;
LABEL_29:
    j_j___libc_free_0_0(v14);
    return a1;
  }
  if ( (*(_QWORD *)(v7 + 8LL * (v8 >> 6)) & (1LL << v8)) != 0 && (unsigned int)sub_C44590(a2 + 16) == v8 )
    goto LABEL_24;
LABEL_5:
  v9 = *(_DWORD *)(a2 + 8);
  if ( !sub_AAF760(a2) && !sub_AB0120(a2) )
  {
    sub_C44830(&v20, a2 + 16, a3);
    sub_C44830(&v18, a2, a3);
    sub_AADC30(a1, (__int64)&v18, &v20);
    sub_969240(&v18);
    sub_969240(&v20);
    return a1;
  }
  v10 = v9 - 1;
  sub_9691E0((__int64)&v18, a3, 0, 0, 0);
  if ( v9 != 1 )
  {
    if ( v10 > 0x40 )
    {
      sub_C43C90(&v18, 0, v10);
    }
    else
    {
      v11 = 0xFFFFFFFFFFFFFFFFLL >> (65 - (unsigned __int8)v9);
      if ( v19 > 0x40 )
        *(_QWORD *)v18 |= v11;
      else
        v18 |= v11;
    }
  }
  sub_C46A40(&v18, 1);
  v12 = v19;
  v19 = 0;
  v21 = v12;
  v20 = v18;
  sub_9691E0((__int64)&v16, a3, 0, 0, 0);
  v13 = v9 + v17 - 1 - a3;
  if ( v17 != (_DWORD)v13 )
  {
    if ( (unsigned int)v13 > 0x3F || v17 > 0x40 )
      sub_C43C90(&v16, v13, v17);
    else
      v16 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v9 - (unsigned __int8)a3 + 63) << (v9
                                                                                        + (unsigned __int8)v17
                                                                                        - 1
                                                                                        - (unsigned __int8)a3);
  }
  sub_AADC30(a1, (__int64)&v16, &v20);
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v19 > 0x40 )
  {
    v14 = v18;
    if ( v18 )
      goto LABEL_29;
  }
  return a1;
}
