// Function: sub_AC3270
// Address: 0xac3270
//
__int64 __fastcall sub_AC3270(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, unsigned int a6)
{
  unsigned int v6; // r12d
  __int64 v7; // r13
  int v8; // r14d
  __int64 *v9; // rcx
  __int64 result; // rax
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rsi

  v6 = a6 & 0x7FFFFFF;
  v7 = 8 * a5;
  v8 = (((a6 & 0x10000000) != 0) << 31) | a6 & 0x7FFFFFF | (((a6 >> 27) & 1) << 30);
  sub_BD35F0(a1, a2, a3);
  v9 = a4;
  result = 32LL * v6;
  *(_DWORD *)(a1 + 4) = *(_DWORD *)(a1 + 4) & 0x38000000 | v8;
  v11 = v7 >> 3;
  if ( v7 > 0 )
  {
    v12 = a1 - result;
    do
    {
      result = *v9;
      if ( *(_QWORD *)v12 )
      {
        v13 = *(_QWORD *)(v12 + 8);
        **(_QWORD **)(v12 + 16) = v13;
        if ( v13 )
          *(_QWORD *)(v13 + 16) = *(_QWORD *)(v12 + 16);
      }
      *(_QWORD *)v12 = result;
      if ( result )
      {
        v14 = *(_QWORD *)(result + 16);
        *(_QWORD *)(v12 + 8) = v14;
        if ( v14 )
          *(_QWORD *)(v14 + 16) = v12 + 8;
        *(_QWORD *)(v12 + 16) = result + 16;
        *(_QWORD *)(result + 16) = v12;
      }
      ++v9;
      v12 += 32;
      --v11;
    }
    while ( v11 );
  }
  return result;
}
