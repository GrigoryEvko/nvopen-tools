// Function: sub_2E91000
// Address: 0x2e91000
//
__int64 __fastcall sub_2E91000(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 **a4, char a5)
{
  _WORD *v5; // r14
  unsigned __int8 *v8; // rsi
  __int64 result; // rax
  unsigned __int64 v11; // rax
  int v12; // ecx
  char v13; // si
  unsigned int v14; // edx
  __int64 *v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rsi
  unsigned __int64 v18; // rcx

  v5 = (_WORD *)a3;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = a3;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  v8 = *a4;
  *(_QWORD *)(a1 + 56) = *a4;
  if ( v8 )
  {
    sub_B976B0((__int64)a4, v8, a1 + 56);
    *a4 = 0;
    a3 = *(_QWORD *)(a1 + 16);
  }
  *(_DWORD *)(a1 + 64) = 0;
  *(_WORD *)(a1 + 68) = *v5;
  result = *(unsigned __int8 *)(a3 + 8) + *(unsigned __int16 *)(a3 + 2) + (unsigned int)*(unsigned __int8 *)(a3 + 9);
  if ( (_DWORD)result )
  {
    v11 = (unsigned int)result - 1LL;
    if ( v11 )
    {
      _BitScanReverse64(&v11, v11);
      v12 = 64 - (v11 ^ 0x3F);
      v13 = 64 - (v11 ^ 0x3F);
      v14 = v12;
      v11 = v12;
    }
    else
    {
      v14 = 0;
      v13 = 0;
      LOBYTE(v12) = 0;
    }
    *(_BYTE *)(a1 + 43) = v13;
    if ( v14 < *(_DWORD *)(a2 + 240) && (v15 = (__int64 *)(*(_QWORD *)(a2 + 232) + 8 * v11), (result = *v15) != 0) )
    {
      *v15 = *(_QWORD *)result;
    }
    else
    {
      v16 = *(_QWORD *)(a2 + 128);
      v17 = 40LL << v12;
      *(_QWORD *)(a2 + 208) += 40LL << v12;
      result = (v16 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      v18 = (40LL << v12) + result;
      if ( *(_QWORD *)(a2 + 136) >= v18 && v16 )
        *(_QWORD *)(a2 + 128) = v18;
      else
        result = sub_9D1E70(a2 + 128, v17, v17, 3);
    }
    *(_QWORD *)(a1 + 32) = result;
  }
  if ( !a5 )
    return sub_2E8F150(a1, a2);
  return result;
}
