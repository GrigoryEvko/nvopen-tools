// Function: sub_2EE9690
// Address: 0x2ee9690
//
char __fastcall sub_2EE9690(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rsi
  __int64 v7; // rdx
  unsigned int v8; // ecx
  char result; // al
  unsigned int v10; // edi

  v3 = *(_QWORD *)(a2 + 24);
  v4 = *(_QWORD *)(a3 + 24);
  if ( v3 == v4 )
    return 1;
  v5 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
  v6 = v5 + 88LL * *(int *)(v3 + 24);
  v7 = v5 + 88LL * *(int *)(v4 + 24);
  v8 = *(_DWORD *)(v6 + 24);
  result = 0;
  if ( v8 != -1 )
  {
    v10 = *(_DWORD *)(v7 + 24);
    if ( v10 != -1 && *(_DWORD *)(v6 + 16) == *(_DWORD *)(v7 + 16) )
      return *(_BYTE *)(v6 + 32) & (v8 <= v10);
  }
  return result;
}
