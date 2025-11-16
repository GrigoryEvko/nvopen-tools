// Function: sub_254F7F0
// Address: 0x254f7f0
//
void __fastcall sub_254F7F0(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rdi
  bool v4; // cc
  unsigned int v5; // eax
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-48h]
  unsigned __int64 v9; // [rsp+10h] [rbp-40h]
  unsigned int v10; // [rsp+18h] [rbp-38h]
  unsigned __int64 v11; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v12; // [rsp+28h] [rbp-28h]
  __int64 v13; // [rsp+30h] [rbp-20h]
  int v14; // [rsp+38h] [rbp-18h]

  sub_AB3510((__int64)&v7, a1 + 16, a2, 0);
  sub_AB2160((__int64)&v11, (__int64)&v7, a1 + 48, 0);
  if ( *(_DWORD *)(a1 + 24) > 0x40u )
  {
    v3 = *(_QWORD *)(a1 + 16);
    if ( v3 )
      j_j___libc_free_0_0(v3);
  }
  v4 = *(_DWORD *)(a1 + 40) <= 0x40u;
  *(_QWORD *)(a1 + 16) = v11;
  v5 = v12;
  v12 = 0;
  *(_DWORD *)(a1 + 24) = v5;
  if ( v4 || (v6 = *(_QWORD *)(a1 + 32)) == 0 )
  {
    *(_QWORD *)(a1 + 32) = v13;
    *(_DWORD *)(a1 + 40) = v14;
    goto LABEL_14;
  }
  j_j___libc_free_0_0(v6);
  v4 = v12 <= 0x40;
  *(_QWORD *)(a1 + 32) = v13;
  *(_DWORD *)(a1 + 40) = v14;
  if ( v4 || !v11 )
  {
LABEL_14:
    if ( v10 <= 0x40 )
      goto LABEL_9;
    goto LABEL_15;
  }
  j_j___libc_free_0_0(v11);
  if ( v10 <= 0x40 )
    goto LABEL_9;
LABEL_15:
  if ( v9 )
    j_j___libc_free_0_0(v9);
LABEL_9:
  if ( v8 > 0x40 )
  {
    if ( v7 )
      j_j___libc_free_0_0(v7);
  }
}
