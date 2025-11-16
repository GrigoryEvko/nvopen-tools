// Function: sub_38D4150
// Address: 0x38d4150
//
__int64 __fastcall sub_38D4150(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // rbx
  __int64 *v6; // rcx
  __int64 *i; // rsi
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 *v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rax

  result = *(unsigned int *)(a1 + 296);
  if ( (_DWORD)result )
  {
    v5 = a2;
    if ( !a2 )
    {
      v9 = sub_22077B0(0xE0u);
      v10 = v9;
      if ( v9 )
      {
        v11 = v9;
        sub_38CF760(v9, 1, 0, 0);
        *(_QWORD *)(v10 + 56) = 0;
        *(_WORD *)(v10 + 48) = 0;
        *(_QWORD *)(v10 + 64) = v10 + 80;
        *(_QWORD *)(v10 + 72) = 0x2000000000LL;
        *(_QWORD *)(v10 + 112) = v10 + 128;
        *(_QWORD *)(v10 + 120) = 0x400000000LL;
      }
      else
      {
        v11 = 0;
      }
      v12 = *(unsigned int *)(a1 + 120);
      if ( (_DWORD)v12 )
        v5 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v12 - 32);
      v13 = *(__int64 **)(a1 + 272);
      v14 = *v13;
      v15 = *(_QWORD *)v10 & 7LL;
      *(_QWORD *)(v10 + 8) = v13;
      v14 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v10 = v14 | v15;
      *(_QWORD *)(v14 + 8) = v11;
      *v13 = *v13 & 7 | v11;
      result = *(unsigned int *)(a1 + 296);
      *(_QWORD *)(v10 + 24) = v5;
      v5 = v10;
    }
    v6 = *(__int64 **)(a1 + 288);
    for ( i = &v6[result]; i != v6; *(_BYTE *)(result + 9) = *(_BYTE *)(result + 9) & 0xF3 | 4 )
    {
      result = *v6++;
      v8 = *(_QWORD *)result;
      *(_QWORD *)(result + 24) = a3;
      *(_QWORD *)result = v5 | v8 & 7;
    }
    *(_DWORD *)(a1 + 296) = 0;
  }
  return result;
}
