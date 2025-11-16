// Function: sub_1E41D50
// Address: 0x1e41d50
//
__int64 __fastcall sub_1E41D50(__int64 a1, __int64 a2)
{
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 result; // rax

  j___libc_free_0(*(_QWORD *)(a1 + 8));
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  ++*(_QWORD *)a1;
  v4 = *(_QWORD *)(a2 + 8);
  ++*(_QWORD *)a2;
  v5 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 8) = v4;
  LODWORD(v4) = *(_DWORD *)(a2 + 16);
  *(_QWORD *)(a2 + 8) = v5;
  LODWORD(v5) = *(_DWORD *)(a1 + 16);
  *(_DWORD *)(a1 + 16) = v4;
  LODWORD(v4) = *(_DWORD *)(a2 + 20);
  *(_DWORD *)(a2 + 16) = v5;
  LODWORD(v5) = *(_DWORD *)(a1 + 20);
  *(_DWORD *)(a1 + 20) = v4;
  LODWORD(v4) = *(_DWORD *)(a2 + 24);
  *(_DWORD *)(a2 + 20) = v5;
  LODWORD(v5) = *(_DWORD *)(a1 + 24);
  *(_DWORD *)(a1 + 24) = v4;
  *(_DWORD *)(a2 + 24) = v5;
  v6 = *(_QWORD *)(a1 + 32);
  v7 = *(_QWORD *)(a1 + 48);
  *(_QWORD *)(a1 + 32) = *(_QWORD *)(a2 + 32);
  *(_QWORD *)(a1 + 40) = *(_QWORD *)(a2 + 40);
  *(_QWORD *)(a1 + 48) = *(_QWORD *)(a2 + 48);
  *(_QWORD *)(a2 + 32) = 0;
  *(_QWORD *)(a2 + 40) = 0;
  *(_QWORD *)(a2 + 48) = 0;
  if ( v6 )
    j_j___libc_free_0(v6, v7 - v6);
  *(_BYTE *)(a1 + 56) = *(_BYTE *)(a2 + 56);
  *(_DWORD *)(a1 + 60) = *(_DWORD *)(a2 + 60);
  *(_DWORD *)(a1 + 64) = *(_DWORD *)(a2 + 64);
  *(_DWORD *)(a1 + 68) = *(_DWORD *)(a2 + 68);
  *(_DWORD *)(a1 + 72) = *(_DWORD *)(a2 + 72);
  *(_QWORD *)(a1 + 80) = *(_QWORD *)(a2 + 80);
  result = *(unsigned int *)(a2 + 88);
  *(_DWORD *)(a1 + 88) = result;
  return result;
}
