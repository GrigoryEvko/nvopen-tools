// Function: sub_27B9960
// Address: 0x27b9960
//
void __fastcall sub_27B9960(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rax

  if ( *(_BYTE *)a1 == 85
    && (v4 = *(_QWORD *)(a1 - 32)) != 0
    && !*(_BYTE *)v4
    && *(_QWORD *)(v4 + 24) == *(_QWORD *)(a1 + 80)
    && (*(_BYTE *)(v4 + 33) & 0x20) != 0 )
  {
    v5 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
    if ( *(_QWORD *)v5 )
    {
      v6 = *(_QWORD *)(v5 + 8);
      **(_QWORD **)(v5 + 16) = v6;
      if ( v6 )
        *(_QWORD *)(v6 + 16) = *(_QWORD *)(v5 + 16);
    }
    *(_QWORD *)v5 = a2;
    if ( a2 )
    {
      v7 = *(_QWORD *)(a2 + 16);
      *(_QWORD *)(v5 + 8) = v7;
      if ( v7 )
        *(_QWORD *)(v7 + 16) = v5 + 8;
      *(_QWORD *)(v5 + 16) = a2 + 16;
      *(_QWORD *)(a2 + 16) = v5;
    }
  }
  else
  {
    if ( *(_QWORD *)(a1 - 96) )
    {
      v2 = *(_QWORD *)(a1 - 88);
      **(_QWORD **)(a1 - 80) = v2;
      if ( v2 )
        *(_QWORD *)(v2 + 16) = *(_QWORD *)(a1 - 80);
    }
    *(_QWORD *)(a1 - 96) = a2;
    if ( a2 )
    {
      v3 = *(_QWORD *)(a2 + 16);
      *(_QWORD *)(a1 - 88) = v3;
      if ( v3 )
        *(_QWORD *)(v3 + 16) = a1 - 88;
      *(_QWORD *)(a1 - 80) = a2 + 16;
      *(_QWORD *)(a2 + 16) = a1 - 96;
    }
  }
}
