// Function: sub_35B8100
// Address: 0x35b8100
//
__int64 __fastcall sub_35B8100(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v3; // rax
  unsigned int v4; // r8d
  __int64 v5; // rdi
  __int64 v6; // rdx
  unsigned int v7; // edx
  __int64 v8; // rsi
  unsigned int v9; // eax

  v2 = *(_QWORD *)(a1 + 16);
  v3 = *(_QWORD *)(a1 + 8);
  v4 = 1;
  v5 = *(_QWORD *)(a2 + 16);
  v6 = *(_QWORD *)(*(_QWORD *)v2 + 24 * v3 + 8);
  v7 = *(_DWORD *)((v6 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v6 >> 1) & 3;
  v8 = *(_QWORD *)(*(_QWORD *)v5 + 24LL * *(_QWORD *)(a2 + 8) + 8);
  v9 = *(_DWORD *)((v8 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v8 >> 1) & 3;
  if ( v7 >= v9 )
  {
    v4 = 0;
    if ( v7 <= v9 )
      LOBYTE(v4) = *(_DWORD *)(v2 + 112) < *(_DWORD *)(v5 + 112);
  }
  return v4;
}
