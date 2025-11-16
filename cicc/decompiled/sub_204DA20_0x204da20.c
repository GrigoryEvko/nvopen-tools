// Function: sub_204DA20
// Address: 0x204da20
//
__int64 __fastcall sub_204DA20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned int *a6)
{
  int v7; // eax
  __int64 result; // rax

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x400000001LL;
  *(_QWORD *)(a1 + 88) = 0x400000001LL;
  *(_QWORD *)(a1 + 16) = a4;
  *(_QWORD *)(a1 + 104) = a1 + 120;
  *(_QWORD *)(a1 + 24) = a5;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_BYTE *)(a1 + 96) = a3;
  *(_QWORD *)(a1 + 112) = 0x400000000LL;
  v7 = *(_DWORD *)(a2 + 8);
  if ( v7 )
  {
    sub_2045020(a1 + 104, a2, a3, a1 + 96, a5, (int)a6);
    v7 = *(_DWORD *)(a2 + 8);
  }
  *(_DWORD *)(a1 + 152) = v7;
  *(_QWORD *)(a1 + 136) = a1 + 152;
  *(_QWORD *)(a1 + 144) = 0x400000001LL;
  result = *((unsigned __int8 *)a6 + 4);
  *(_BYTE *)(a1 + 172) = result;
  if ( (_BYTE)result )
  {
    result = *a6;
    *(_DWORD *)(a1 + 168) = result;
  }
  return result;
}
