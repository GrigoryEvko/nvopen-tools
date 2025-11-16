// Function: sub_254F8E0
// Address: 0x254f8e0
//
void __fastcall sub_254F8E0(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rdi
  bool v4; // cc
  unsigned int v5; // eax
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned int v8; // eax
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-38h]
  __int64 v12; // [rsp+10h] [rbp-30h]
  int v13; // [rsp+18h] [rbp-28h]

  sub_AB2160((__int64)&v10, a1 + 16, a2, 0);
  if ( *(_DWORD *)(a1 + 24) > 0x40u )
  {
    v3 = *(_QWORD *)(a1 + 16);
    if ( v3 )
      j_j___libc_free_0_0(v3);
  }
  v4 = *(_DWORD *)(a1 + 40) <= 0x40u;
  *(_QWORD *)(a1 + 16) = v10;
  v5 = v11;
  v11 = 0;
  *(_DWORD *)(a1 + 24) = v5;
  if ( v4 || (v6 = *(_QWORD *)(a1 + 32)) == 0 )
  {
    *(_QWORD *)(a1 + 32) = v12;
    *(_DWORD *)(a1 + 40) = v13;
  }
  else
  {
    j_j___libc_free_0_0(v6);
    v4 = v11 <= 0x40;
    *(_QWORD *)(a1 + 32) = v12;
    *(_DWORD *)(a1 + 40) = v13;
    if ( !v4 && v10 )
      j_j___libc_free_0_0(v10);
  }
  sub_AB2160((__int64)&v10, a1 + 48, a2, 0);
  if ( *(_DWORD *)(a1 + 56) > 0x40u )
  {
    v7 = *(_QWORD *)(a1 + 48);
    if ( v7 )
      j_j___libc_free_0_0(v7);
  }
  v4 = *(_DWORD *)(a1 + 72) <= 0x40u;
  *(_QWORD *)(a1 + 48) = v10;
  v8 = v11;
  v11 = 0;
  *(_DWORD *)(a1 + 56) = v8;
  if ( v4 || (v9 = *(_QWORD *)(a1 + 64)) == 0 )
  {
    *(_QWORD *)(a1 + 64) = v12;
    *(_DWORD *)(a1 + 72) = v13;
  }
  else
  {
    j_j___libc_free_0_0(v9);
    v4 = v11 <= 0x40;
    *(_QWORD *)(a1 + 64) = v12;
    *(_DWORD *)(a1 + 72) = v13;
    if ( !v4 )
    {
      if ( v10 )
        j_j___libc_free_0_0(v10);
    }
  }
}
