// Function: sub_2F8FB10
// Address: 0x2f8fb10
//
__int64 __fastcall sub_2F8FB10(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // r9
  __int64 v5; // rax
  int v6; // edx
  int v7; // ecx
  unsigned __int64 v8; // rcx
  __int64 result; // rax
  int v10; // edx
  unsigned __int64 v11; // r12
  _DWORD v12[9]; // [rsp+Ch] [rbp-24h] BYREF

  v3 = a1 + 320;
  v12[0] = (__int64)(*(_QWORD *)(v3 - 16) - *(_QWORD *)(v3 - 24)) >> 2;
  sub_2F8E5C0(v3, v12);
  v12[0] = *(_DWORD *)(a2 + 200);
  sub_2F8E5C0(a1 + 296, v12);
  v5 = (__int64)(*(_QWORD *)(a1 + 328) - *(_QWORD *)(a1 + 320)) >> 2;
  LOBYTE(v6) = v5;
  v7 = *(_DWORD *)(a1 + 408) & 0x3F;
  if ( v7 )
    *(_QWORD *)(*(_QWORD *)(a1 + 344) + 8LL * *(unsigned int *)(a1 + 352) - 8) &= ~(-1LL << v7);
  *(_DWORD *)(a1 + 408) = v5;
  v8 = *(unsigned int *)(a1 + 352);
  result = (unsigned int)(v5 + 63) >> 6;
  if ( (unsigned int)result != v8 )
  {
    if ( (unsigned int)result >= v8 )
    {
      v11 = (unsigned int)result - v8;
      if ( (unsigned int)result > (unsigned __int64)*(unsigned int *)(a1 + 356) )
      {
        sub_C8D5F0(a1 + 344, (const void *)(a1 + 360), (unsigned int)result, 8u, (unsigned int)result, v4);
        v8 = *(unsigned int *)(a1 + 352);
      }
      result = *(_QWORD *)(a1 + 344);
      if ( 8 * v11 )
      {
        result = (__int64)memset((void *)(result + 8 * v8), 0, 8 * v11);
        LODWORD(v8) = *(_DWORD *)(a1 + 352);
      }
      v6 = *(_DWORD *)(a1 + 408);
      *(_DWORD *)(a1 + 352) = v11 + v8;
    }
    else
    {
      *(_DWORD *)(a1 + 352) = result;
    }
  }
  v10 = v6 & 0x3F;
  if ( v10 )
  {
    result = ~(-1LL << v10);
    *(_QWORD *)(*(_QWORD *)(a1 + 344) + 8LL * *(unsigned int *)(a1 + 352) - 8) &= result;
  }
  return result;
}
