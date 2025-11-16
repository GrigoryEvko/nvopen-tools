// Function: sub_2FC8970
// Address: 0x2fc8970
//
__int64 __fastcall sub_2FC8970(__int64 a1)
{
  unsigned int v1; // eax

  v1 = sub_2FC8910(a1);
  if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 32LL) + 40LL * v1 + 24) )
    return v1 + 1;
  else
    return 0xFFFFFFFFLL;
}
