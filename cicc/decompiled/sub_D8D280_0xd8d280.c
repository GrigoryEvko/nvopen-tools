// Function: sub_D8D280
// Address: 0xd8d280
//
__int64 __fastcall sub_D8D280(__int64 *a1, __int64 *a2)
{
  __int64 v3; // rax
  int v4; // edx
  int v5; // r13d
  __int64 v6; // r15
  __int64 v7; // rcx
  __int64 v8; // r14
  __int64 v9; // rdi
  __int64 result; // rax
  bool v11; // cc
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // [rsp+0h] [rbp-70h]
  __int64 v15; // [rsp+0h] [rbp-70h]
  int v16; // [rsp+Ch] [rbp-64h]
  int v17; // [rsp+Ch] [rbp-64h]
  __int64 v18; // [rsp+18h] [rbp-58h]

  v3 = a1[1];
  v4 = *((_DWORD *)a1 + 6);
  *((_DWORD *)a1 + 6) = 0;
  v5 = *((_DWORD *)a1 + 10);
  v6 = *a1;
  *((_DWORD *)a1 + 10) = 0;
  v7 = a1[2];
  v8 = a1[4];
  v18 = v3;
  *a1 = *a2;
  a1[1] = a2[1];
  a1[2] = a2[2];
  *((_DWORD *)a1 + 6) = *((_DWORD *)a2 + 6);
  *((_DWORD *)a2 + 6) = 0;
  if ( *((_DWORD *)a1 + 10) > 0x40u )
  {
    v9 = a1[4];
    if ( v9 )
    {
      v14 = v7;
      v16 = v4;
      j_j___libc_free_0_0(v9);
      v7 = v14;
      v4 = v16;
    }
  }
  a1[4] = a2[4];
  *((_DWORD *)a1 + 10) = *((_DWORD *)a2 + 10);
  result = v18;
  v11 = *((_DWORD *)a2 + 6) <= 0x40u;
  *((_DWORD *)a2 + 10) = 0;
  *a2 = v6;
  a2[1] = v18;
  if ( v11 || (v12 = a2[2]) == 0 )
  {
    a2[2] = v7;
    *((_DWORD *)a2 + 6) = v4;
  }
  else
  {
    v15 = v7;
    v17 = v4;
    result = j_j___libc_free_0_0(v12);
    v11 = *((_DWORD *)a2 + 10) <= 0x40u;
    a2[2] = v15;
    *((_DWORD *)a2 + 6) = v17;
    if ( !v11 )
    {
      v13 = a2[4];
      if ( v13 )
        result = j_j___libc_free_0_0(v13);
    }
  }
  a2[4] = v8;
  *((_DWORD *)a2 + 10) = v5;
  return result;
}
