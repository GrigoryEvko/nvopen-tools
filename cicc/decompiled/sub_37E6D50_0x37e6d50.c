// Function: sub_37E6D50
// Address: 0x37e6d50
//
__int64 __fastcall sub_37E6D50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int i; // r8d
  __int64 v7; // rsi
  __int64 v8; // rdx
  unsigned __int8 v9; // cl
  unsigned __int8 v10; // r12
  int v11; // r14d
  int v12; // edi

  result = *(_QWORD *)(a2 + 8);
  for ( i = *(_DWORD *)(a2 + 24); result != a3; result = *(_QWORD *)(result + 8) )
  {
    v7 = *(_QWORD *)(a1 + 200);
    v8 = i;
    v9 = *(_BYTE *)(result + 208);
    i = *(_DWORD *)(result + 24);
    v10 = *(_BYTE *)(*(_QWORD *)(result + 32) + 340LL);
    v11 = *(_DWORD *)(v7 + 8 * v8 + 4) + *(_DWORD *)(v7 + 8 * v8);
    v12 = -(1 << v9) & ((1 << v9) + v11 - 1);
    if ( v9 > v10 )
      v12 = (-(1 << v9) & ((1 << v9) + v11 - 1)) - (1LL << v10) + (1LL << v9);
    *(_DWORD *)(v7 + 8LL * i) = v12;
  }
  return result;
}
