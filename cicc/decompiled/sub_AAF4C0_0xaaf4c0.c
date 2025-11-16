// Function: sub_AAF4C0
// Address: 0xaaf4c0
//
__int64 __fastcall sub_AAF4C0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // ebx
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // r15
  char v8; // al
  unsigned int v9; // r15d
  __int64 v11; // rax
  unsigned int v12; // eax
  unsigned int v13; // eax
  unsigned int v15; // eax
  char v17; // [rsp+8h] [rbp-68h]
  __int64 v18; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-58h]
  __int64 v20; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v21; // [rsp+28h] [rbp-48h]
  __int64 v22; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v23; // [rsp+38h] [rbp-38h]

  v4 = *(_DWORD *)(a2 + 8);
  v21 = v4;
  if ( v4 <= 0x40 )
  {
    v5 = *(_QWORD *)a2;
LABEL_3:
    v6 = *(_QWORD *)a3;
    v7 = *(_QWORD *)a3 & v5;
LABEL_4:
    v8 = v7 == v6;
    goto LABEL_5;
  }
  sub_C43780(&v20, a2);
  if ( v21 <= 0x40 )
  {
    v5 = v20;
    goto LABEL_3;
  }
  sub_C43B90(&v20, a3);
  v13 = v21;
  v7 = v20;
  v21 = 0;
  v23 = v13;
  v22 = v20;
  if ( v13 <= 0x40 )
  {
    v6 = *(_QWORD *)a3;
    goto LABEL_4;
  }
  v8 = sub_C43C50(&v22, a3);
  if ( v7 )
  {
    v17 = v8;
    j_j___libc_free_0_0(v7);
    v8 = v17;
    if ( v21 > 0x40 )
    {
      if ( v20 )
      {
        j_j___libc_free_0_0(v20);
        v8 = v17;
      }
    }
  }
LABEL_5:
  if ( !v8 )
  {
    sub_AADB10(a1, v4, 1);
    return a1;
  }
  v9 = *(_DWORD *)(a2 + 8);
  if ( v9 <= 0x40 )
  {
    if ( !*(_QWORD *)a2 )
      goto LABEL_8;
  }
  else if ( v9 == (unsigned int)sub_C444A0(a2) )
  {
LABEL_8:
    sub_AADB10(a1, v4, 0);
    return a1;
  }
  v23 = *(_DWORD *)(a3 + 8);
  if ( v23 > 0x40 )
  {
    sub_C43780(&v22, a3);
    v9 = *(_DWORD *)(a2 + 8);
  }
  else
  {
    v22 = *(_QWORD *)a3;
  }
  if ( v9 <= 0x40 )
  {
    _RDX = *(_QWORD *)a2;
    v15 = 64;
    __asm { tzcnt   rcx, rdx }
    if ( *(_QWORD *)a2 )
      v15 = _RCX;
    if ( v9 > v15 )
      v9 = v15;
  }
  else
  {
    v9 = sub_C44590(a2);
  }
  sub_9691E0((__int64)&v18, v4, 0, 0, 0);
  v11 = 1LL << v9;
  if ( v19 <= 0x40 )
    v18 |= v11;
  else
    *(_QWORD *)(v18 + 8LL * (v9 >> 6)) |= v11;
  sub_C45EE0(&v18, a3);
  v12 = v19;
  v19 = 0;
  v21 = v12;
  v20 = v18;
  sub_9875E0(a1, &v20, &v22);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  return a1;
}
