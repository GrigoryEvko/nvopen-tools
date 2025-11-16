// Function: sub_1600160
// Address: 0x1600160
//
__int64 __fastcall sub_1600160(__int64 a1, __int64 a2, int a3)
{
  unsigned int v3; // eax
  __int64 v5; // rsi
  __int64 result; // rax
  _QWORD *v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // rax

  v3 = *(_DWORD *)(a1 + 20) & 0xF0000000 | 1;
  v5 = (unsigned int)(a3 + 1);
  *(_DWORD *)(a1 + 56) = v5;
  *(_DWORD *)(a1 + 20) = v3;
  result = sub_1648880(a1, v5, 0);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
  {
    v7 = *(_QWORD **)(a1 - 8);
  }
  else
  {
    result = 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
    v7 = (_QWORD *)(a1 - result);
  }
  if ( *v7 )
  {
    v8 = v7[1];
    result = v7[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)result = v8;
    if ( v8 )
    {
      result |= *(_QWORD *)(v8 + 16) & 3LL;
      *(_QWORD *)(v8 + 16) = result;
    }
  }
  *v7 = a2;
  if ( a2 )
  {
    v9 = *(_QWORD *)(a2 + 8);
    v7[1] = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = (unsigned __int64)(v7 + 1) | *(_QWORD *)(v9 + 16) & 3LL;
    result = (a2 + 8) | v7[2] & 3LL;
    v7[2] = result;
    *(_QWORD *)(a2 + 8) = v7;
  }
  return result;
}
