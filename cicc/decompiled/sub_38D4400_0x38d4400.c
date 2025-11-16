// Function: sub_38D4400
// Address: 0x38d4400
//
__int64 __fastcall sub_38D4400(__int64 a1, unsigned int a2, __int64 a3, int a4, unsigned int a5)
{
  unsigned int v5; // r15d
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 *v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 result; // rax

  v5 = a5;
  if ( !a5 )
    v5 = a2;
  v7 = sub_22077B0(0x48u);
  v8 = v7;
  if ( v7 )
  {
    v9 = v7;
    sub_38CF760(v7, 0, 0, 0);
    *(_BYTE *)(v8 + 52) &= ~1u;
    *(_DWORD *)(v8 + 48) = a2;
    *(_QWORD *)(v8 + 56) = a3;
    *(_DWORD *)(v8 + 68) = v5;
    *(_DWORD *)(v8 + 64) = a4;
  }
  else
  {
    v9 = 0;
  }
  sub_38D4150(a1, v8, 0);
  v10 = *(unsigned int *)(a1 + 120);
  v11 = 0;
  if ( (_DWORD)v10 )
    v11 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v10 - 32);
  v12 = *(__int64 **)(a1 + 272);
  v13 = *v12;
  v14 = *(_QWORD *)v8 & 7LL;
  *(_QWORD *)(v8 + 8) = v12;
  v13 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v8 = v13 | v14;
  *(_QWORD *)(v13 + 8) = v9;
  *v12 = *v12 & 7 | v9;
  v15 = *(unsigned int *)(a1 + 120);
  *(_QWORD *)(v8 + 24) = v11;
  if ( !(_DWORD)v15 )
    BUG();
  result = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v15 - 32);
  if ( a2 > *(_DWORD *)(result + 24) )
    *(_DWORD *)(result + 24) = a2;
  return result;
}
