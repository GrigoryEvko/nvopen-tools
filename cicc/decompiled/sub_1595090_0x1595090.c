// Function: sub_1595090
// Address: 0x1595090
//
__int64 __fastcall sub_1595090(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v10; // rcx
  unsigned __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rbx
  __int64 *v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rdi
  unsigned __int64 v18; // rsi
  __int64 v19; // rsi

  sub_1648CB0(a1, a6, 5);
  *(_QWORD *)(a1 + 24) = a2;
  *(_DWORD *)(a1 + 20) = (a5 + 1) & 0xFFFFFFF | *(_DWORD *)(a1 + 20) & 0xF0000000;
  *(_WORD *)(a1 + 18) = 32;
  *(_QWORD *)(a1 + 32) = sub_15FA030(a2, a4, a5);
  result = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  if ( *(_QWORD *)result )
  {
    v10 = *(_QWORD *)(result + 8);
    v11 = *(_QWORD *)(result + 16) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v11 = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = *(_QWORD *)(v10 + 16) & 3LL | v11;
  }
  *(_QWORD *)result = a3;
  if ( a3 )
  {
    v12 = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(result + 8) = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = (result + 8) | *(_QWORD *)(v12 + 16) & 3LL;
    v13 = *(_QWORD *)(result + 16);
    *(_QWORD *)(a3 + 8) = result;
    *(_QWORD *)(result + 16) = (a3 + 8) | v13 & 3;
  }
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
  {
    v14 = *(_QWORD *)(a1 - 8);
  }
  else
  {
    result = 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
    v14 = a1 - result;
  }
  if ( (_DWORD)a5 )
  {
    v15 = a4;
    result = v14 + 32;
    do
    {
      v16 = *v15;
      if ( *(_QWORD *)(result - 8) )
      {
        v17 = *(_QWORD *)result;
        v18 = *(_QWORD *)(result + 8) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v18 = *(_QWORD *)result;
        if ( v17 )
          *(_QWORD *)(v17 + 16) = *(_QWORD *)(v17 + 16) & 3LL | v18;
      }
      *(_QWORD *)(result - 8) = v16;
      if ( v16 )
      {
        v19 = *(_QWORD *)(v16 + 8);
        *(_QWORD *)result = v19;
        if ( v19 )
          *(_QWORD *)(v19 + 16) = result | *(_QWORD *)(v19 + 16) & 3LL;
        *(_QWORD *)(result + 8) = (v16 + 8) | *(_QWORD *)(result + 8) & 3LL;
        *(_QWORD *)(v16 + 8) = result - 8;
      }
      ++v15;
      result += 24;
    }
    while ( &a4[(unsigned int)(a5 - 1) + 1] != v15 );
  }
  return result;
}
