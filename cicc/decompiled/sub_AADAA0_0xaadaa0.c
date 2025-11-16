// Function: sub_AADAA0
// Address: 0xaadaa0
//
__int64 __fastcall sub_AADAA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rdx
  unsigned __int64 v6; // rax
  int v7; // eax

  v5 = *(unsigned int *)(a2 + 8);
  if ( (unsigned int)v5 > 0x40 )
  {
    sub_C43D10(a2, a2, v5, a4, a5);
  }
  else
  {
    v6 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v5) & ~*(_QWORD *)a2;
    if ( !(_DWORD)v5 )
      v6 = 0;
    *(_QWORD *)a2 = v6;
  }
  sub_C46250(a2);
  v7 = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a2 + 8) = 0;
  *(_DWORD *)(a1 + 8) = v7;
  *(_QWORD *)a1 = *(_QWORD *)a2;
  return a1;
}
