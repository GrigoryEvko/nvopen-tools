// Function: sub_2A8BDE0
// Address: 0x2a8bde0
//
__int64 __fastcall sub_2A8BDE0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r12
  __int64 v4; // rdx
  unsigned __int64 v5; // r13
  __int64 v6; // r9
  int v7; // eax
  unsigned __int64 v8; // rdi
  unsigned int v9; // eax
  __int64 result; // rax
  __int64 v11; // r14
  unsigned __int64 v12; // r12
  int v13; // r15d
  int v14; // r15d
  unsigned __int64 v15[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = a2;
  v4 = *(unsigned int *)(a1 + 8);
  v5 = *(_QWORD *)a1;
  v6 = v4 + 1;
  v7 = *(_DWORD *)(a1 + 8);
  if ( v4 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    v11 = a1 + 16;
    if ( v5 > a2 || a2 >= v5 + 24 * v4 )
    {
      v5 = sub_C8D7D0(a1, a1 + 16, v4 + 1, 0x18u, v15, v6);
      sub_2A8AC40((__int64 *)a1, v5);
      v14 = v15[0];
      if ( *(_QWORD *)a1 != v11 )
        _libc_free(*(_QWORD *)a1);
      v4 = *(unsigned int *)(a1 + 8);
      *(_QWORD *)a1 = v5;
      *(_DWORD *)(a1 + 12) = v14;
      v7 = v4;
    }
    else
    {
      v12 = a2 - v5;
      v5 = sub_C8D7D0(a1, a1 + 16, v4 + 1, 0x18u, v15, v6);
      sub_2A8AC40((__int64 *)a1, v5);
      v13 = v15[0];
      if ( *(_QWORD *)a1 != v11 )
        _libc_free(*(_QWORD *)a1);
      v4 = *(unsigned int *)(a1 + 8);
      *(_QWORD *)a1 = v5;
      v2 = v5 + v12;
      *(_DWORD *)(a1 + 12) = v13;
      v7 = v4;
    }
  }
  v8 = v5 + 24 * v4;
  if ( v8 )
  {
    *(_QWORD *)v8 = *(_QWORD *)v2;
    v9 = *(_DWORD *)(v2 + 16);
    *(_DWORD *)(v8 + 16) = v9;
    if ( v9 > 0x40 )
      sub_C43780(v8 + 8, (const void **)(v2 + 8));
    else
      *(_QWORD *)(v8 + 8) = *(_QWORD *)(v2 + 8);
    v7 = *(_DWORD *)(a1 + 8);
  }
  result = (unsigned int)(v7 + 1);
  *(_DWORD *)(a1 + 8) = result;
  return result;
}
