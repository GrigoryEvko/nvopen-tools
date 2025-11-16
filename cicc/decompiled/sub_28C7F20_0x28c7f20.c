// Function: sub_28C7F20
// Address: 0x28c7f20
//
__int64 __fastcall sub_28C7F20(__int64 a1, __int64 a2)
{
  unsigned int v3; // eax
  __int64 v4; // rdi
  __int64 v5; // r12
  unsigned int *v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  _QWORD v9[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( *(_DWORD *)(a1 + 12) != *(_DWORD *)(a2 + 12) )
    return 0;
  if ( *(_QWORD *)(a1 + 40) != *(_QWORD *)(a2 + 40) )
    return 0;
  v3 = *(_DWORD *)(a1 + 36);
  if ( v3 != *(_DWORD *)(a2 + 36) || 8LL * v3 && memcmp(*(const void **)(a1 + 24), *(const void **)(a2 + 24), 8LL * v3) )
    return 0;
  if ( *(_QWORD *)(a1 + 48) != *(_QWORD *)(a2 + 48) || *(_DWORD *)(a2 + 8) != 10 )
    return 0;
  v4 = *(_QWORD *)(a1 + 56);
  v9[0] = *(_QWORD *)(v4 + 72);
  v5 = *(_QWORD *)(*(_QWORD *)(a2 + 56) + 72LL);
  v6 = (unsigned int *)sub_BD5C60(v4);
  v7 = sub_A7AD50(v9, v6, v5);
  v9[2] = v8;
  v9[1] = v7;
  return (unsigned __int8)v8;
}
