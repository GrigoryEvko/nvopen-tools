// Function: sub_17ADA40
// Address: 0x17ada40
//
__int64 __fastcall sub_17ADA40(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, unsigned int a6)
{
  unsigned int v7; // r8d
  __int64 v8; // rax
  __int64 *v9; // rbx
  unsigned __int64 v10; // r12
  __int64 result; // rax
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  const void *v16; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-38h]

  v7 = a6;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v8 = *(_QWORD *)(a2 - 8);
  else
    v8 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v9 = (__int64 *)(v8 + 24LL * a3);
  v17 = *(_DWORD *)(a4 + 8);
  if ( v17 > 0x40 )
  {
    sub_16A4FD0((__int64)&v16, (const void **)a4);
    v7 = a6;
  }
  else
  {
    v16 = *(const void **)a4;
  }
  v10 = sub_17A9010(a1, *v9, (__int64)&v16, a5, v7, a2);
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  result = 0;
  if ( v10 )
  {
    if ( *v9 )
    {
      v12 = v9[1];
      v13 = v9[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v13 = v12;
      if ( v12 )
        *(_QWORD *)(v12 + 16) = *(_QWORD *)(v12 + 16) & 3LL | v13;
    }
    *v9 = v10;
    v14 = *(_QWORD *)(v10 + 8);
    v9[1] = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = (unsigned __int64)(v9 + 1) | *(_QWORD *)(v14 + 16) & 3LL;
    v9[2] = v9[2] & 3 | (v10 + 8);
    result = 1;
    *(_QWORD *)(v10 + 8) = v9;
  }
  return result;
}
