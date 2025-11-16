// Function: sub_30EC150
// Address: 0x30ec150
//
__int64 __fastcall sub_30EC150(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 i; // rax

  v3 = *(_QWORD *)(a3 + 80);
  while ( a3 + 72 != v3 )
  {
    v4 = v3;
    v3 = *(_QWORD *)(v3 + 8);
    v5 = v4 + 24;
    for ( i = *(_QWORD *)(v4 + 32); v5 != i; i = *(_QWORD *)(i + 8) )
    {
      if ( (unsigned int)*(unsigned __int8 *)(i - 24) - 30 > 0x42 )
        BUG();
    }
  }
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
