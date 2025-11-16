// Function: sub_2A45070
// Address: 0x2a45070
//
__int64 __fastcall sub_2A45070(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbp
  __int64 v4; // rax
  unsigned int v5; // r8d
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v10; // r10
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14[3]; // [rsp-18h] [rbp-18h] BYREF

  v4 = *(unsigned int *)(a2 + 8);
  v5 = 0;
  if ( !(_DWORD)v4 )
    return v5;
  v6 = *(_QWORD *)a2 + 48 * v4 - 48;
  v5 = *(unsigned __int8 *)(v6 + 40);
  if ( !(_BYTE)v5 )
  {
    if ( *(_DWORD *)a3 >= *(_DWORD *)v6 )
    {
      LOBYTE(v5) = *(_DWORD *)(a3 + 4) <= *(_DWORD *)(v6 + 4);
      return v5;
    }
    return v5;
  }
  v7 = *(_QWORD *)(a3 + 24);
  if ( !v7 )
    return 0;
  v8 = *(_QWORD *)(v7 + 24);
  v5 = 0;
  if ( *(_BYTE *)v8 != 84 )
    return v5;
  v10 = *(_QWORD *)(v6 + 32);
  v11 = *(_QWORD *)(v10 + 56);
  if ( v11 != *(_QWORD *)(*(_QWORD *)(v8 - 8)
                        + 32LL * *(unsigned int *)(v8 + 72)
                        + 8LL * (unsigned int)((v7 - *(_QWORD *)(v8 - 8)) >> 5)) )
    return v5;
  v14[2] = v3;
  v12 = *(_QWORD *)(v10 + 64);
  v13 = *(_QWORD *)(a1 + 16);
  v14[0] = v11;
  v14[1] = v12;
  return sub_B19ED0(v13, v14, v7);
}
