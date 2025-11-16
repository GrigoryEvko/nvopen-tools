// Function: sub_1399480
// Address: 0x1399480
//
__int64 __fastcall sub_1399480(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rbx
  __int64 i; // r12
  unsigned __int64 v5; // rsi

  *(_QWORD *)a1 = a2;
  *(_DWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = a1 + 16;
  *(_QWORD *)(a1 + 40) = a1 + 16;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = sub_1399010((_QWORD *)a1, 0);
  result = sub_22077B0(40);
  if ( result )
  {
    *(_QWORD *)result = 0;
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)(result + 16) = 0;
    *(_QWORD *)(result + 24) = 0;
    *(_DWORD *)(result + 32) = 0;
  }
  *(_QWORD *)(a1 + 64) = result;
  v3 = *(_QWORD *)(a2 + 32);
  for ( i = a2 + 24; i != v3; v3 = *(_QWORD *)(v3 + 8) )
  {
    v5 = v3 - 56;
    if ( !v3 )
      v5 = 0;
    result = sub_1399160((_QWORD *)a1, v5);
  }
  return result;
}
