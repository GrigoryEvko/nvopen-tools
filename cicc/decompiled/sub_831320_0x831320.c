// Function: sub_831320
// Address: 0x831320
//
__int64 __fastcall sub_831320(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rax
  bool v5; // cl
  int v6; // ecx
  int v7; // esi

  *(_QWORD *)a3 = a2;
  *(_QWORD *)(a3 + 24) = 0xFFFFFFFFLL;
  *(_QWORD *)(a3 + 8) = 0;
  result = *(_QWORD *)&dword_4F077C8;
  *(_QWORD *)(a3 + 16) = 0;
  *(_QWORD *)(a3 + 32) = 0;
  *(_QWORD *)(a3 + 40) = 0;
  *(_QWORD *)(a3 + 48) = 0;
  *(_DWORD *)(a3 + 56) = 0;
  *(_QWORD *)(a3 + 64) = 0;
  *(_QWORD *)(a3 + 72) = 0;
  *(_QWORD *)(a3 + 80) = result;
  if ( a1 )
  {
    while ( *(_BYTE *)(a1 + 140) == 12 )
      a1 = *(_QWORD *)(a1 + 160);
    v4 = *(_QWORD *)(a1 + 168);
    *(_QWORD *)(a3 + 8) = *(_QWORD *)v4;
    v5 = (*(_BYTE *)(v4 + 16) & 2) != 0;
    *(_BYTE *)(a3 + 19) = v5;
    *(_BYTE *)(a3 + 20) = *(_BYTE *)(v4 + 16) & 1;
    if ( !v5 )
      v5 = *(_QWORD *)(v4 + 8) != 0;
    *(_BYTE *)(a3 + 18) = v5;
    *(_BYTE *)(a3 + 23) = *(_BYTE *)(v4 + 24);
    *(_DWORD *)(a3 + 24) = *(__int16 *)(v4 + 22);
    v6 = *(_DWORD *)(v4 + 28);
    *(_DWORD *)(a3 + 48) = v6;
    v7 = *(_DWORD *)(v4 + 32);
    *(_DWORD *)(a3 + 52) = v7;
    if ( *(_QWORD *)(v4 + 40) )
    {
      *(_DWORD *)(a3 + 48) = v6 - 1;
      if ( v7 )
        *(_DWORD *)(a3 + 52) = v7 - 1;
    }
    result = *(unsigned int *)(v4 + 36);
    *(_DWORD *)(a3 + 56) = result;
  }
  return result;
}
