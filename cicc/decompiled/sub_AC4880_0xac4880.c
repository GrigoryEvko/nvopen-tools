// Function: sub_AC4880
// Address: 0xac4880
//
__int64 __fastcall sub_AC4880(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        unsigned int a8)
{
  int v9; // r15d
  __int64 v10; // rax
  int v11; // eax
  int v12; // eax
  __int64 v13; // rax
  __int64 result; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // r12
  __int64 *v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 v21; // rsi

  sub_BD35F0(a1, a6, 5);
  *(_WORD *)(a1 + 2) = 34;
  v9 = *(_DWORD *)(a1 + 4);
  *(_QWORD *)(a1 + 24) = a2;
  *(_DWORD *)(a1 + 4) = v9 & 0x38000000 | (((a8 >> 27) & 1) << 30) | a8 & 0x7FFFFFF | (a8 >> 28 << 31);
  v10 = sub_B4DCA0(a2, a4, a5);
  *(_BYTE *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 32) = v10;
  if ( *(_BYTE *)(a7 + 32) )
  {
    v11 = *(_DWORD *)(a7 + 8);
    *(_DWORD *)(a7 + 8) = 0;
    *(_DWORD *)(a1 + 48) = v11;
    *(_QWORD *)(a1 + 40) = *(_QWORD *)a7;
    v12 = *(_DWORD *)(a7 + 24);
    *(_DWORD *)(a7 + 24) = 0;
    *(_DWORD *)(a1 + 64) = v12;
    v13 = *(_QWORD *)(a7 + 16);
    *(_BYTE *)(a1 + 72) = 1;
    *(_QWORD *)(a1 + 56) = v13;
  }
  result = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( *(_QWORD *)result )
  {
    v15 = *(_QWORD *)(result + 8);
    **(_QWORD **)(result + 16) = v15;
    if ( v15 )
      *(_QWORD *)(v15 + 16) = *(_QWORD *)(result + 16);
  }
  *(_QWORD *)result = a3;
  if ( a3 )
  {
    v16 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(result + 8) = v16;
    if ( v16 )
      *(_QWORD *)(v16 + 16) = result + 8;
    *(_QWORD *)(result + 16) = a3 + 16;
    *(_QWORD *)(a3 + 16) = result;
  }
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    v17 = *(_QWORD *)(a1 - 8);
  }
  else
  {
    result = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
    v17 = a1 - result;
  }
  if ( (_DWORD)a5 )
  {
    v18 = a4;
    result = v17 + 32;
    do
    {
      v19 = *v18;
      if ( *(_QWORD *)result )
      {
        v20 = *(_QWORD *)(result + 8);
        **(_QWORD **)(result + 16) = v20;
        if ( v20 )
          *(_QWORD *)(v20 + 16) = *(_QWORD *)(result + 16);
      }
      *(_QWORD *)result = v19;
      if ( v19 )
      {
        v21 = *(_QWORD *)(v19 + 16);
        *(_QWORD *)(result + 8) = v21;
        if ( v21 )
          *(_QWORD *)(v21 + 16) = result + 8;
        *(_QWORD *)(result + 16) = v19 + 16;
        *(_QWORD *)(v19 + 16) = result;
      }
      result += 32;
      ++v18;
    }
    while ( result != v17 + 32LL * (unsigned int)(a5 - 1) + 64 );
  }
  return result;
}
