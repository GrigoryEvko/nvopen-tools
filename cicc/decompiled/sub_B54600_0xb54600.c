// Function: sub_B54600
// Address: 0xb54600
//
__int64 __fastcall sub_B54600(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // rsi
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // rdx

  v4 = (unsigned int)(a3 + 1);
  *(_DWORD *)(a1 + 4) = *(_DWORD *)(a1 + 4) & 0xF8000000 | 1;
  *(_DWORD *)(a1 + 72) = v4;
  sub_BD2A10(a1, v4, 0);
  result = *(_QWORD *)(a1 - 8);
  if ( *(_QWORD *)result )
  {
    v6 = *(_QWORD *)(result + 8);
    **(_QWORD **)(result + 16) = v6;
    if ( v6 )
      *(_QWORD *)(v6 + 16) = *(_QWORD *)(result + 16);
  }
  *(_QWORD *)result = a2;
  if ( a2 )
  {
    v7 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(result + 8) = v7;
    if ( v7 )
      *(_QWORD *)(v7 + 16) = result + 8;
    *(_QWORD *)(result + 16) = a2 + 16;
    *(_QWORD *)(a2 + 16) = result;
  }
  return result;
}
