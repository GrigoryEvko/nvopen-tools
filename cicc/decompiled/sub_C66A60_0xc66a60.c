// Function: sub_C66A60
// Address: 0xc66a60
//
__int64 __fastcall sub_C66A60(__int64 a1, int a2)
{
  int v2; // ebx
  __int64 v3; // rsi

  sub_C66990(a1, *(unsigned __int8 **)(a1 + 16), *(_QWORD *)(a1 + 32) - *(_QWORD *)(a1 + 16));
  sub_C66990(a1, *(unsigned __int8 **)(a1 + 16), *(_QWORD *)(a1 + 32) - *(_QWORD *)(a1 + 16));
  v2 = a2 - *(_DWORD *)(a1 + 56);
  v3 = (unsigned int)v2;
  if ( v2 <= 0 )
    v3 = 1;
  sub_CB69B0(a1, v3);
  return a1;
}
