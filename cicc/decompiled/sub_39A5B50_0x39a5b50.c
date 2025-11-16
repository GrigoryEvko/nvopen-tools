// Function: sub_39A5B50
// Address: 0x39a5b50
//
__int64 __fastcall sub_39A5B50(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r13
  __int64 v8; // r15
  __int64 v9; // rdx
  unsigned __int64 v10; // rsi
  __int64 result; // rax
  __int64 v12; // rcx
  __int64 v13; // r14
  unsigned __int64 v14; // rcx
  unsigned int v15; // edi
  __int64 *v16; // r14
  __int64 v17; // rsi
  unsigned __int8 *v18; // rsi
  _DWORD v19[13]; // [rsp+Ch] [rbp-34h] BYREF

  v7 = sub_39A5A90(a1, 33, a2, 0);
  sub_39A3B20(a1, v7, 73, a4);
  v8 = *(_QWORD *)(a3 + 24);
  v9 = sub_39A2270(a1);
  v10 = *(_QWORD *)(a3 - 8LL * *(unsigned int *)(a3 + 8));
  result = *(unsigned __int8 *)v10;
  if ( (_BYTE)result == 1 )
  {
    v12 = *(_QWORD *)(v10 + 136);
  }
  else
  {
    v13 = -1;
    if ( (unsigned int)(unsigned __int8)result - 24 > 1 )
      goto LABEL_7;
    v12 = v10 | 4;
  }
  v13 = -1;
  if ( (v12 & 4) == 0 )
  {
    v14 = v12 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v14 )
    {
      v15 = *(_DWORD *)(v14 + 32);
      v16 = *(__int64 **)(v14 + 24);
      if ( v15 > 0x40 )
        v13 = *v16;
      else
        v13 = (__int64)((_QWORD)v16 << (64 - (unsigned __int8)v15)) >> (64 - (unsigned __int8)v15);
    }
  }
LABEL_7:
  if ( v9 == -1 || v9 != v8 )
  {
    v19[0] = 65551;
    sub_39A3560(a1, (__int64 *)(v7 + 8), 34, (__int64)v19, v8);
    v10 = *(_QWORD *)(a3 - 8LL * *(unsigned int *)(a3 + 8));
    result = *(unsigned __int8 *)v10;
  }
  if ( (_BYTE)result == 1 )
  {
    v17 = *(_QWORD *)(v10 + 136);
  }
  else
  {
    result = (unsigned int)(result - 24);
    if ( (unsigned int)result > 1 )
      goto LABEL_15;
    v17 = v10 | 4;
  }
  if ( (v17 & 4) != 0 )
  {
    v18 = (unsigned __int8 *)(v17 & 0xFFFFFFFFFFFFFFF8LL);
    if ( v18 )
    {
      result = (__int64)sub_39A23D0(a1, v18);
      if ( result )
        return sub_39A3B20(a1, v7, 55, result);
      return result;
    }
  }
LABEL_15:
  if ( v13 != -1 )
  {
    BYTE2(v19[0]) = 0;
    return sub_39A3560(a1, (__int64 *)(v7 + 8), 55, (__int64)v19, v13);
  }
  return result;
}
