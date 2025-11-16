// Function: sub_106EF90
// Address: 0x106ef90
//
__int64 __fastcall sub_106EF90(__int64 a1, __int64 a2, __int64 a3)
{
  const void *v3; // r13
  size_t v4; // r12
  int v5; // eax
  int v6; // eax
  __int64 v7; // r12
  int v8; // eax
  _QWORD v10[3]; // [rsp+0h] [rbp-60h] BYREF
  __int64 v11[2]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v12; // [rsp+30h] [rbp-30h] BYREF

  v10[0] = a2;
  v10[1] = a3;
  sub_C93130(v11, (__int64)v10);
  v3 = (const void *)v11[0];
  v4 = v11[1];
  v5 = sub_C92610();
  v6 = sub_C92860((__int64 *)(a1 + 440), v3, v4, v5);
  if ( v6 == -1 )
    v7 = *(_QWORD *)(a1 + 440) + 8LL * *(unsigned int *)(a1 + 448);
  else
    v7 = *(_QWORD *)(a1 + 440) + 8LL * v6;
  if ( (__int64 *)v11[0] != &v12 )
    j_j___libc_free_0(v11[0], v12 + 1);
  if ( v7 == *(_QWORD *)(a1 + 440) + 8LL * *(unsigned int *)(a1 + 448) )
    return 0;
  v8 = *(_DWORD *)(*(_QWORD *)v7 + 8LL);
  BYTE4(v11[0]) = 1;
  LODWORD(v11[0]) = v8;
  return v11[0];
}
