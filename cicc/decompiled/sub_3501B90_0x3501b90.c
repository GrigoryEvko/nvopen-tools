// Function: sub_3501B90
// Address: 0x3501b90
//
bool __fastcall sub_3501B90(unsigned int *a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // r11d
  __int64 v5; // rcx
  __int16 *v6; // rax
  unsigned int v7; // esi
  int v8; // edx
  int v9; // r8d

  v4 = a1[14];
  v5 = 0;
  v6 = (__int16 *)(*(_QWORD *)(a3 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(a3 + 8) + 24LL * *a1 + 16) >> 12));
  v7 = *(_DWORD *)(*(_QWORD *)(a3 + 8) + 24LL * *a1 + 16) & 0xFFF;
  v8 = 0;
  while ( 1 )
  {
    if ( !v6 )
      return v8 == v4;
    if ( v8 == v4 || *(_DWORD *)(*((_QWORD *)a1 + 6) + v5 + 88) != *(_DWORD *)(a2 + 216LL * v7) )
      break;
    v9 = *v6;
    ++v8;
    ++v6;
    v5 += 112;
    v7 += v9;
    if ( !(_WORD)v9 )
      return v8 == v4;
  }
  return 0;
}
