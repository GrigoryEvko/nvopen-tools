// Function: sub_2332320
// Address: 0x2332320
//
unsigned __int64 __fastcall sub_2332320(__int64 a1, char a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 result; // rax
  unsigned int v7; // r13d
  unsigned int v8; // edx
  unsigned __int64 v9; // r8
  int v10; // edx
  unsigned __int64 v11; // r14
  int v12; // edx

  result = *(unsigned int *)(a1 + 8);
  v7 = *(_DWORD *)(a1 + 64);
  v8 = v7 + 1;
  if ( v7 + 1 <= *(_DWORD *)(a1 + 8) << 6 )
  {
    *(_DWORD *)(a1 + 64) = v8;
    goto LABEL_3;
  }
  if ( (v7 & 0x3F) != 0 )
  {
    *(_QWORD *)(*(_QWORD *)a1 + 8 * result - 8) &= ~(-1LL << (v7 & 0x3F));
    result = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 64) = v8;
  v9 = (v7 + 64) >> 6;
  if ( v9 == result )
  {
LABEL_11:
    v10 = v8 & 0x3F;
    if ( !v10 )
      goto LABEL_3;
    goto LABEL_12;
  }
  if ( v9 < result )
  {
    *(_DWORD *)(a1 + 8) = (v7 + 64) >> 6;
    goto LABEL_11;
  }
  v11 = v9 - result;
  if ( v9 > *(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), (v7 + 64) >> 6, 8u, v9, a6);
    result = *(unsigned int *)(a1 + 8);
  }
  if ( 8 * v11 )
  {
    memset((void *)(*(_QWORD *)a1 + 8 * result), 0, 8 * v11);
    result = *(unsigned int *)(a1 + 8);
  }
  v12 = *(_DWORD *)(a1 + 64);
  result += v11;
  *(_DWORD *)(a1 + 8) = result;
  v10 = v12 & 0x3F;
  if ( v10 )
  {
LABEL_12:
    result = ~(-1LL << v10);
    *(_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 8) &= result;
  }
LABEL_3:
  if ( a2 )
  {
    result = v7 >> 6;
    *(_QWORD *)(*(_QWORD *)a1 + 8 * result) |= 1LL << v7;
  }
  return result;
}
