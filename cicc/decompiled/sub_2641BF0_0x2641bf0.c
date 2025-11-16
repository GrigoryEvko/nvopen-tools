// Function: sub_2641BF0
// Address: 0x2641bf0
//
__int64 __fastcall sub_2641BF0(__int64 a1, __int64 a2)
{
  unsigned __int64 v4; // rdi
  __int64 v5; // rsi
  __int64 v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 result; // rax

  *(_QWORD *)a1 = *(_QWORD *)a2;
  v4 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 8) = *(_QWORD *)(a2 + 8);
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 24);
  *(_QWORD *)(a2 + 8) = 0;
  *(_QWORD *)(a2 + 16) = 0;
  *(_QWORD *)(a2 + 24) = 0;
  if ( v4 )
    j_j___libc_free_0(v4);
  v5 = *(unsigned int *)(a1 + 64);
  v6 = *(_QWORD *)(a1 + 48);
  *(_QWORD *)(a1 + 32) = *(_QWORD *)(a2 + 32);
  sub_C7D6A0(v6, 4 * v5, 4);
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  ++*(_QWORD *)(a1 + 40);
  v7 = *(_QWORD *)(a2 + 48);
  ++*(_QWORD *)(a2 + 40);
  v8 = *(_QWORD *)(a1 + 48);
  *(_QWORD *)(a1 + 48) = v7;
  LODWORD(v7) = *(_DWORD *)(a2 + 56);
  *(_QWORD *)(a2 + 48) = v8;
  LODWORD(v8) = *(_DWORD *)(a1 + 56);
  *(_DWORD *)(a1 + 56) = v7;
  LODWORD(v7) = *(_DWORD *)(a2 + 60);
  *(_DWORD *)(a2 + 56) = v8;
  LODWORD(v8) = *(_DWORD *)(a1 + 60);
  *(_DWORD *)(a1 + 60) = v7;
  LODWORD(v7) = *(_DWORD *)(a2 + 64);
  *(_DWORD *)(a2 + 60) = v8;
  result = *(unsigned int *)(a1 + 64);
  *(_DWORD *)(a1 + 64) = v7;
  *(_DWORD *)(a2 + 64) = result;
  return result;
}
