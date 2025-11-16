// Function: sub_3368DA0
// Address: 0x3368da0
//
__int64 __fastcall sub_3368DA0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 result; // rax
  __int64 v5; // rax

  *(_QWORD *)(a1 + 8) = a3;
  *(_BYTE *)(a1 + 20) = a4;
  *(_QWORD *)a1 = a2 & 0xFFFFFFFFFFFFFFFBLL;
  result = 0;
  if ( a2 )
  {
    v5 = *(_QWORD *)(a2 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
      v5 = **(_QWORD **)(v5 + 16);
    result = *(_DWORD *)(v5 + 8) >> 8;
  }
  *(_DWORD *)(a1 + 16) = result;
  return result;
}
