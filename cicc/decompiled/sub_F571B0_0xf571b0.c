// Function: sub_F571B0
// Address: 0xf571b0
//
__int64 __fastcall sub_F571B0(__int64 a1, __int64 a2)
{
  unsigned int v3; // r8d
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rcx

  v3 = 0;
  v4 = *(_QWORD *)(a1 + 40);
  v5 = *(_QWORD *)(a1 + 16);
  while ( v5 )
  {
    v6 = v5;
    v5 = *(_QWORD *)(v5 + 8);
    if ( v4 != *(_QWORD *)(*(_QWORD *)(v6 + 24) + 40LL) )
    {
      if ( *(_QWORD *)v6 )
      {
        **(_QWORD **)(v6 + 16) = v5;
        if ( v5 )
          *(_QWORD *)(v5 + 16) = *(_QWORD *)(v6 + 16);
      }
      *(_QWORD *)v6 = a2;
      if ( a2 )
      {
        v7 = *(_QWORD *)(a2 + 16);
        *(_QWORD *)(v6 + 8) = v7;
        if ( v7 )
          *(_QWORD *)(v7 + 16) = v6 + 8;
        *(_QWORD *)(v6 + 16) = a2 + 16;
        *(_QWORD *)(a2 + 16) = v6;
      }
      ++v3;
    }
  }
  return v3;
}
