// Function: sub_2DF4670
// Address: 0x2df4670
//
void __fastcall sub_2DF4670(__int64 a1, int a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 i; // rdx
  __int64 v6; // rsi

  if ( a2 )
  {
    v4 = *(_QWORD *)(a1 + 8);
    for ( i = 16LL * (unsigned int)(a2 - 1); i; i -= 16 )
    {
      *(_QWORD *)(*(_QWORD *)(i + v4) + 8LL * *(unsigned int *)(i + v4 + 12) + 96) = a3;
      v4 = *(_QWORD *)(a1 + 8);
      v6 = v4 + i;
      if ( *(_DWORD *)(v6 + 8) - 1 != *(_DWORD *)(v6 + 12) )
        return;
    }
    *(_QWORD *)(*(_QWORD *)v4 + 8LL * *(unsigned int *)(v4 + 12) + 72) = a3;
  }
}
