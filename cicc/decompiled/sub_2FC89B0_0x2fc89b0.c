// Function: sub_2FC89B0
// Address: 0x2fc89b0
//
__int64 __fastcall sub_2FC89B0(__int64 a1)
{
  unsigned int v2; // eax
  __int64 v3; // rdi
  unsigned int v4; // esi
  __int64 v5; // r12
  int i; // r12d

  v2 = sub_2FC8910(a1);
  v3 = *(_QWORD *)a1;
  v4 = v2 + 1;
  v5 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 32LL) + 40LL * v2 + 24);
  if ( (_DWORD)v5 )
  {
    for ( i = v5 - 2; ; --i )
    {
      v4 = sub_2FC88B0(v3, v4);
      if ( i == -1 )
        break;
      v3 = *(_QWORD *)a1;
    }
  }
  return v4 + 1;
}
