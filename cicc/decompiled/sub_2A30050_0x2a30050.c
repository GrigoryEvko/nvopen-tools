// Function: sub_2A30050
// Address: 0x2a30050
//
__int64 __fastcall sub_2A30050(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // rbx

  v3 = a1 + 32;
  v4 = a1 + 80;
  if ( a3 + 56 == (*(_QWORD *)(a3 + 56) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    *(_QWORD *)(a1 + 8) = v3;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 56) = v4;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  else
  {
    sub_2A3FF50(a3, 0, 0);
    memset((void *)a1, 0, 0x60u);
    *(_QWORD *)(a1 + 8) = v3;
    *(_QWORD *)(a1 + 56) = v4;
    *(_DWORD *)(a1 + 16) = 2;
    *(_BYTE *)(a1 + 28) = 1;
    *(_DWORD *)(a1 + 64) = 2;
    *(_BYTE *)(a1 + 76) = 1;
    return a1;
  }
}
