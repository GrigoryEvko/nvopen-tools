// Function: sub_140EC20
// Address: 0x140ec20
//
__int64 __fastcall sub_140EC20(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD **v4; // rax
  unsigned int v6; // eax
  unsigned int v7; // r13d
  __int64 v8; // rbx
  __int64 v9; // rax
  bool v10; // cc
  __int64 v11; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v12; // [rsp+8h] [rbp-68h]
  __int64 v13; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-58h]
  __int64 v15; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v16; // [rsp+28h] [rbp-48h]
  __int64 v17; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v18; // [rsp+38h] [rbp-38h]

  if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
    v4 = *(_QWORD ***)(a3 - 8);
  else
    v4 = (_QWORD **)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
  sub_140E6D0((__int64)&v15, a2, *v4);
  v12 = *(_DWORD *)(a2 + 20);
  if ( v12 > 0x40 )
    sub_16A4EF0(&v11, 0, 0);
  else
    v11 = 0;
  if ( v16 > 1 && v18 > 1 && (unsigned __int8)sub_1634900(a3, *(_QWORD *)a2, &v11) )
  {
    v14 = v18;
    if ( v18 > 0x40 )
      sub_16A4FD0(&v13, &v17);
    else
      v13 = v17;
    sub_16A7200(&v13, &v11);
    v6 = v16;
    v7 = v14;
    v14 = 0;
    v8 = v13;
    *(_DWORD *)(a1 + 8) = v16;
    if ( v6 > 0x40 )
    {
      sub_16A4FD0(a1, &v15);
      v10 = v14 <= 0x40;
      *(_DWORD *)(a1 + 24) = v7;
      *(_QWORD *)(a1 + 16) = v8;
      if ( !v10 && v13 )
        j_j___libc_free_0_0(v13);
    }
    else
    {
      v9 = v15;
      *(_DWORD *)(a1 + 24) = v7;
      *(_QWORD *)(a1 + 16) = v8;
      *(_QWORD *)a1 = v9;
    }
  }
  else
  {
    *(_DWORD *)(a1 + 8) = 1;
    *(_QWORD *)a1 = 0;
    *(_DWORD *)(a1 + 24) = 1;
    *(_QWORD *)(a1 + 16) = 0;
  }
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0(v11);
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  return a1;
}
