// Function: sub_1D95130
// Address: 0x1d95130
//
__int64 __fastcall sub_1D95130(__int64 a1)
{
  int v1; // r12d
  _DWORD *v3; // rdx
  _DWORD *v4; // rsi
  __int64 v5; // r14
  int v6; // r9d
  int v7; // eax
  __int64 v8; // r8
  __int64 v9; // rcx
  bool v10; // zf
  __int64 v11; // rdx
  __int64 v12; // rsi
  unsigned int *v13; // r10
  __int64 v14; // r9
  __int64 v15; // rdi
  __int64 v16; // r15
  int v17; // r13d
  __int64 v18; // r10
  __int64 (*v19)(); // rax
  int v20; // eax
  _BYTE *v21; // rsi
  unsigned int v22; // r12d
  _BYTE *v24; // rsi
  int v25; // eax

  v3 = *(_DWORD **)(a1 + 40);
  v4 = *(_DWORD **)(a1 + 8);
  v5 = (unsigned int)v3[3];
  v6 = v3[1] + v3[2];
  v7 = **(_DWORD **)(a1 + 16) + **(_DWORD **)(a1 + 24);
  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 16LL);
  v9 = (unsigned int)v4[3];
  v10 = v4[1] + v4[2] == v7;
  v11 = (unsigned int)(v4[1] + v4[2] - v7);
  v12 = *(_QWORD *)a1;
  v13 = *(unsigned int **)(a1 + 48);
  LOBYTE(v1) = !v10;
  v14 = (unsigned int)(v6 - v7);
  v15 = *(_QWORD *)(a1 + 56);
  v17 = v1;
  v16 = *v13;
  LOBYTE(v17) = (_DWORD)v14 != 0 && !v10;
  if ( (_BYTE)v17 )
  {
    v18 = *(_QWORD *)(v15 + 544);
    v17 = 0;
    v19 = *(__int64 (**)())(*(_QWORD *)v18 + 336LL);
    if ( v19 != sub_1D91880 )
    {
      v25 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64, __int64, __int64, __int64, __int64, __int64))v19)(
              v18,
              *(_QWORD *)(v12 + 16),
              v11,
              v9,
              v8,
              v14,
              v5,
              v16);
      v15 = *(_QWORD *)(a1 + 56);
      v12 = *(_QWORD *)a1;
      v17 = v25;
    }
  }
  if ( (*(_BYTE *)v12 & 1) != 0 )
  {
    v24 = *(_BYTE **)(a1 + 32);
    v22 = *v24 & 1;
    if ( (*v24 & 1) == 0 )
    {
      sub_1D94D00(v15, (__int64)v24, *(_QWORD *)(a1 + 72), 0, 0, 1u);
      return v22;
    }
    return 0;
  }
  v20 = sub_1D94D00(v15, v12, *(_QWORD *)(a1 + 64) + 40LL, 0, 0, 1u);
  v21 = *(_BYTE **)(a1 + 32);
  if ( (*v21 & 1) != 0 )
    return 0;
  return (unsigned int)sub_1D94D00(*(_QWORD *)(a1 + 56), (__int64)v21, *(_QWORD *)(a1 + 72), 0, 0, 1u) & v17 & v20;
}
