// Function: sub_1E1C450
// Address: 0x1e1c450
//
void __fastcall sub_1E1C450(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 **a4, char a5)
{
  unsigned __int8 *v8; // rsi
  __int64 v9; // rdi
  _WORD *v11; // rcx
  unsigned int v12; // esi
  int v13; // eax
  _WORD *v14; // rcx
  int v15; // eax
  __int64 v16; // rax
  unsigned __int8 v17; // si
  unsigned __int64 v18; // rax

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = a3;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 40) = 0;
  *(_BYTE *)(a1 + 44) = 0;
  *(_DWORD *)(a1 + 46) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  v8 = *a4;
  *(_QWORD *)(a1 + 64) = *a4;
  v9 = a3;
  if ( v8 )
  {
    sub_1623210((__int64)a4, v8, a1 + 64);
    *a4 = 0;
    v9 = *(_QWORD *)(a1 + 16);
  }
  v11 = *(_WORD **)(v9 + 32);
  v12 = *(unsigned __int16 *)(v9 + 2);
  if ( v11 && *v11 )
  {
    v13 = 0;
    do
      ++v13;
    while ( v11[v13] );
    v12 += v13;
  }
  v14 = *(_WORD **)(v9 + 24);
  if ( v14 && *v14 )
  {
    v15 = 0;
    do
      ++v15;
    while ( v14[v15] );
    v12 += v15;
  }
  if ( v12 )
  {
    v16 = v12;
    v17 = 0;
    v18 = v16 - 1;
    if ( v18 )
    {
      _BitScanReverse64(&v18, v18);
      v17 = 64 - (v18 ^ 0x3F);
    }
    *(_BYTE *)(a1 + 44) = v17;
    *(_QWORD *)(a1 + 32) = sub_1E1A7D0(a2 + 232, v17, (__int64 *)(a2 + 120));
  }
  if ( !a5 )
    sub_1E1AEE0(a1, a2);
}
