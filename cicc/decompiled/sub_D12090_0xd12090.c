// Function: sub_D12090
// Address: 0xd12090
//
__int64 __fastcall sub_D12090(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 i; // rbx

  *(_QWORD *)a1 = a2;
  *(_DWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = a1 + 16;
  *(_QWORD *)(a1 + 40) = a1 + 16;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = sub_D110B0((_QWORD *)a1, 0);
  result = sub_22077B0(48);
  if ( result )
  {
    *(_QWORD *)result = a1;
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)(result + 16) = 0;
    *(_QWORD *)(result + 24) = 0;
    *(_QWORD *)(result + 32) = 0;
    *(_DWORD *)(result + 40) = 0;
  }
  *(_QWORD *)(a1 + 64) = result;
  for ( i = *(_QWORD *)(a2 + 32); a2 + 24 != i; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    result = (unsigned int)(*(_DWORD *)(i - 20) - 68);
    if ( (unsigned int)result > 3 )
      result = sub_D11F30((_QWORD *)a1, i - 56);
  }
  return result;
}
