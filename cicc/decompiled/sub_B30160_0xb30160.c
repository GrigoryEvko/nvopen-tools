// Function: sub_B30160
// Address: 0xb30160
//
char __fastcall sub_B30160(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax

  if ( a2 )
  {
    if ( sub_B2FC80(a1) )
      *(_DWORD *)(a1 + 4) = *(_DWORD *)(a1 + 4) & 0xF8000000 | 1;
    if ( *(_QWORD *)(a1 - 32) )
    {
      v2 = *(_QWORD *)(a1 - 24);
      **(_QWORD **)(a1 - 16) = v2;
      if ( v2 )
        *(_QWORD *)(v2 + 16) = *(_QWORD *)(a1 - 16);
    }
    *(_QWORD *)(a1 - 32) = a2;
    v3 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(a1 - 24) = v3;
    if ( v3 )
      *(_QWORD *)(v3 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = a2 + 16;
    *(_QWORD *)(a2 + 16) = a1 - 32;
  }
  else
  {
    LOBYTE(v3) = sub_B2FC80(a1);
    if ( !(_BYTE)v3 )
    {
      if ( *(_QWORD *)(a1 - 32) )
      {
        v3 = *(_QWORD *)(a1 - 24);
        **(_QWORD **)(a1 - 16) = v3;
        if ( v3 )
          *(_QWORD *)(v3 + 16) = *(_QWORD *)(a1 - 16);
      }
      *(_DWORD *)(a1 + 4) &= 0xF8000000;
      *(_QWORD *)(a1 - 32) = 0;
    }
  }
  return v3;
}
