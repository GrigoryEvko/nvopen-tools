// Function: sub_19048B0
// Address: 0x19048b0
//
__int64 __fastcall sub_19048B0(__int64 a1, __int64 a2, __int64 *a3)
{
  unsigned int v3; // ecx
  __int64 v4; // rax

  v3 = *((_DWORD *)a3 + 2);
  if ( dword_4FAE5E0 + 1 < v3 )
  {
    sub_1904850(a1);
  }
  else
  {
    v4 = *a3;
    *(_DWORD *)(a1 + 8) = v3;
    *((_DWORD *)a3 + 2) = 0;
    *(_QWORD *)a1 = v4;
    LODWORD(v4) = *((_DWORD *)a3 + 6);
    *((_DWORD *)a3 + 6) = 0;
    *(_DWORD *)(a1 + 24) = v4;
    *(_QWORD *)(a1 + 16) = a3[2];
  }
  return a1;
}
