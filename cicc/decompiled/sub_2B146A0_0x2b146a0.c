// Function: sub_2B146A0
// Address: 0x2b146a0
//
__int64 __fastcall sub_2B146A0(__int64 a1)
{
  __int64 *v1; // r13
  __int64 v2; // rax
  __int64 v3; // rcx

  v1 = *(__int64 **)(*(_QWORD *)a1 + 3296LL);
  sub_DFBCC0(**(unsigned __int8 ***)(a1 + 16));
  v2 = **(_QWORD **)(a1 + 16);
  if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
    v3 = *(_QWORD *)(v2 - 8);
  else
    v3 = v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
  return sub_DFD060(v1, **(unsigned int **)(a1 + 8), *(_QWORD *)(v2 + 8), *(_QWORD *)(*(_QWORD *)v3 + 8LL));
}
