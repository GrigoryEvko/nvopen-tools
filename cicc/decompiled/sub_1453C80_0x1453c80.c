// Function: sub_1453C80
// Address: 0x1453c80
//
__int64 __fastcall sub_1453C80(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 v6; // rdx

  result = sub_157F280(**(_QWORD **)(a1 + 32));
  if ( v3 != result )
  {
    v4 = result;
    result = *(unsigned int *)(a2 + 8);
    v5 = v3;
    do
    {
      if ( *(_DWORD *)(a2 + 12) <= (unsigned int)result )
      {
        sub_16CD150(a2, a2 + 16, 0, 8);
        result = *(unsigned int *)(a2 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8 * result) = v4;
      result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = result;
      if ( !v4 )
        BUG();
      v6 = *(_QWORD *)(v4 + 32);
      if ( !v6 )
        BUG();
      v4 = 0;
      if ( *(_BYTE *)(v6 - 8) == 77 )
        v4 = v6 - 24;
    }
    while ( v5 != v4 );
  }
  return result;
}
