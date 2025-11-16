// Function: sub_16E8E60
// Address: 0x16e8e60
//
__int64 __fastcall sub_16E8E60(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // rdi
  __int64 v5; // r10
  __int64 v6; // r9
  __int64 v7; // rax
  __int64 result; // rax

  v2 = *(_QWORD *)(a1 + 56);
  v4 = *(int *)(v2 + 16);
  v5 = *(_QWORD *)(v2 + 24);
  v6 = 32LL * *(int *)(v2 + 20);
  if ( *(_DWORD *)(v2 + 16) )
  {
    v7 = 0;
    do
    {
      *(_BYTE *)(*(_QWORD *)a2 + (unsigned __int8)v7) &= ~*(_BYTE *)(a2 + 8);
      *(_BYTE *)(a2 + 9) -= v7++;
    }
    while ( v4 != v7 );
  }
  result = v5 + v6 - 32;
  if ( a2 == result )
  {
    result = *(_QWORD *)(a1 + 56);
    --*(_DWORD *)(result + 20);
  }
  return result;
}
