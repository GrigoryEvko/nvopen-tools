// Function: sub_15A6B80
// Address: 0x15a6b80
//
void __fastcall sub_15A6B80(__int64 a1, __int64 a2)
{
  unsigned int v3; // eax
  _QWORD *v4; // rdi

  if ( a2 && (*(_BYTE *)(a2 + 1) == 2 || *(_DWORD *)(a2 + 12)) )
  {
    v3 = *(_DWORD *)(a1 + 352);
    if ( v3 >= *(_DWORD *)(a1 + 356) )
    {
      sub_15A6A10(a1 + 344, 0);
      v3 = *(_DWORD *)(a1 + 352);
    }
    v4 = (_QWORD *)(*(_QWORD *)(a1 + 344) + 8LL * v3);
    if ( v4 )
    {
      *v4 = a2;
      sub_1623A60(v4, a2, 2);
      v3 = *(_DWORD *)(a1 + 352);
    }
    *(_DWORD *)(a1 + 352) = v3 + 1;
  }
}
