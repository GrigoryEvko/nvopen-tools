// Function: sub_772F00
// Address: 0x772f00
//
__int64 __fastcall sub_772F00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // r8
  __int64 v8; // r13
  int v9; // eax
  unsigned int v10; // r9d
  __int64 v11; // rdi
  unsigned int i; // eax
  __int64 *v13; // rdx
  int v14; // eax
  __int64 v16; // rsi

  v5 = *(_QWORD *)(a1 + 16);
  v6 = qword_4F080A8;
  if ( (unsigned int)(0x10000 - (*(_DWORD *)(a1 + 16) - *(_DWORD *)(a1 + 24))) <= 0x3F )
  {
    sub_772E70((_QWORD *)(a1 + 16));
    v6 = qword_4F080A8;
    v5 = *(_QWORD *)(a1 + 16);
  }
  v7 = a1 + 72;
  *(_QWORD *)(a1 + 16) = v5 + 64;
  v8 = v5 + 16;
  *(_QWORD *)v5 = 0;
  *(_OWORD *)(v5 + 24) = 0;
  *(_QWORD *)(v5 + 8) = v6;
  *(_QWORD *)(v5 + 16) = a2;
  *(_QWORD *)(v5 + 40) = a3;
  v9 = *(_DWORD *)(a1 + 128);
  *(_BYTE *)(v5 + 7) |= 1u;
  *(_DWORD *)(v5 + 28) = v9;
  *(_DWORD *)(v5 + 48) = *(_DWORD *)(a1 + 128);
  v10 = *(_DWORD *)(a1 + 8);
  v11 = *(_QWORD *)a1;
  for ( i = v10 & ((unsigned __int64)(a1 + 72) >> 3); ; i = v10 & (i + 1) )
  {
    v13 = (__int64 *)(v11 + 16LL * i);
    if ( !*v13 )
      break;
    if ( v7 == *v13 )
    {
      v16 = v11 + 16LL * i;
      *(_QWORD *)(v5 + 56) = *(_QWORD *)(v16 + 8);
      *(_QWORD *)(v16 + 8) = v8;
      return v5 + 16;
    }
  }
  *v13 = v7;
  v13[1] = v8;
  v14 = *(_DWORD *)(a1 + 12) + 1;
  *(_DWORD *)(a1 + 12) = v14;
  if ( 2 * v14 > v10 )
    sub_7704A0(a1);
  *(_QWORD *)(v5 + 56) = 0;
  return v5 + 16;
}
