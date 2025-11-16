// Function: sub_34B8A40
// Address: 0x34b8a40
//
__int64 __fastcall sub_34B8A40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rax
  unsigned int v9; // r8d
  __int64 v11; // r8
  __int64 v12; // r9
  unsigned int v13[13]; // [rsp+1Ch] [rbp-34h] BYREF

  while ( 1 )
  {
    v13[0] = 0;
    v7 = sub_B501B0(a1, v13, 1);
    if ( !v7 )
      break;
    v5 = *(unsigned int *)(a2 + 8);
    if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      sub_C8D5F0(a2, (const void *)(a2 + 16), v5 + 1, 8u, v11, v12);
      v5 = *(unsigned int *)(a2 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a2 + 8 * v5) = a1;
    ++*(_DWORD *)(a2 + 8);
    v6 = *(unsigned int *)(a3 + 8);
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
      sub_C8D5F0(a3, (const void *)(a3 + 16), v6 + 1, 4u, v11, v12);
      v6 = *(unsigned int *)(a3 + 8);
    }
    a1 = v7;
    *(_DWORD *)(*(_QWORD *)a3 + 4 * v6) = 0;
    ++*(_DWORD *)(a3 + 8);
  }
  v8 = *(unsigned int *)(a3 + 8);
  v9 = 1;
  if ( (_DWORD)v8 )
  {
    while ( (unsigned __int8)(*(_BYTE *)(sub_B501B0(
                                           *(_QWORD *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8) - 8),
                                           (unsigned int *)(*(_QWORD *)a3 + 4 * v8 - 4),
                                           1)
                                       + 8)
                            - 15) <= 1u )
    {
      v9 = sub_34B8410(a2, a3);
      if ( !(_BYTE)v9 )
        return v9;
      v8 = *(unsigned int *)(a3 + 8);
    }
    return 1;
  }
  return v9;
}
