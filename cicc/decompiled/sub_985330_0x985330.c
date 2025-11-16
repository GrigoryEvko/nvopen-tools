// Function: sub_985330
// Address: 0x985330
//
__int64 __fastcall sub_985330(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 result; // rax
  __int64 v4; // rdi
  __int64 v5; // rbx
  __int64 v6; // rdi
  __int64 v7; // [rsp+0h] [rbp-80h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-78h]
  __int64 v9; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v10; // [rsp+18h] [rbp-68h]
  __int64 v11; // [rsp+20h] [rbp-60h]
  unsigned int v12; // [rsp+28h] [rbp-58h]
  __int64 v13; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v14; // [rsp+38h] [rbp-48h]
  __int64 v15; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v16; // [rsp+48h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 16);
  if ( **(_DWORD **)a1 == **(_DWORD **)(a1 + 8) )
  {
    sub_C70430(&v13, 1, 0, 0, v2, a2);
    v5 = *(_QWORD *)(a1 + 16);
    if ( *(_DWORD *)(v5 + 8) > 0x40u && *(_QWORD *)v5 )
      j_j___libc_free_0_0(*(_QWORD *)v5);
    *(_QWORD *)v5 = v13;
    *(_DWORD *)(v5 + 8) = v14;
    v14 = 0;
    if ( *(_DWORD *)(v5 + 24) <= 0x40u || (v6 = *(_QWORD *)(v5 + 16)) == 0 )
    {
      *(_QWORD *)(v5 + 16) = v15;
      result = v16;
      *(_DWORD *)(v5 + 24) = v16;
      return result;
    }
    j_j___libc_free_0_0(v6);
    result = v14;
    *(_QWORD *)(v5 + 16) = v15;
    *(_DWORD *)(v5 + 24) = v16;
    if ( (unsigned int)result > 0x40 )
    {
      v4 = v13;
      if ( v13 )
        return j_j___libc_free_0_0(v4);
    }
  }
  else
  {
    sub_C44740(&v13, v2 + 16);
    sub_C44740(&v7, v2);
    v10 = v8;
    v9 = v7;
    v12 = v14;
    v11 = v13;
    sub_C70430(&v13, 1, 0, 0, &v9, a2);
    sub_C43D80(v2, &v13, 0);
    result = sub_C43D80(v2 + 16, &v15, 0);
    if ( v16 > 0x40 && v15 )
      result = j_j___libc_free_0_0(v15);
    if ( v14 > 0x40 && v13 )
      result = j_j___libc_free_0_0(v13);
    if ( v12 > 0x40 && v11 )
      result = j_j___libc_free_0_0(v11);
    if ( v10 > 0x40 )
    {
      v4 = v9;
      if ( v9 )
        return j_j___libc_free_0_0(v4);
    }
  }
  return result;
}
