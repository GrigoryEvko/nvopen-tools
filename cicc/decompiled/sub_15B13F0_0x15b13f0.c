// Function: sub_15B13F0
// Address: 0x15b13f0
//
void __fastcall sub_15B13F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax

  if ( a2 > 0 )
  {
    v2 = *(unsigned int *)(a1 + 8);
    if ( (unsigned int)v2 >= *(_DWORD *)(a1 + 12) )
    {
      sub_16CD150(a1, a1 + 16, 0, 8);
      v2 = *(unsigned int *)(a1 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a1 + 8 * v2) = 35;
    v3 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
    *(_DWORD *)(a1 + 8) = v3;
    if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v3 )
    {
      sub_16CD150(a1, a1 + 16, 0, 8);
      v3 = *(unsigned int *)(a1 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a1 + 8 * v3) = a2;
    ++*(_DWORD *)(a1 + 8);
  }
  else if ( a2 )
  {
    v4 = *(unsigned int *)(a1 + 8);
    if ( (unsigned int)v4 >= *(_DWORD *)(a1 + 12) )
    {
      sub_16CD150(a1, a1 + 16, 0, 8);
      v4 = *(unsigned int *)(a1 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a1 + 8 * v4) = 16;
    v5 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
    *(_DWORD *)(a1 + 8) = v5;
    if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v5 )
    {
      sub_16CD150(a1, a1 + 16, 0, 8);
      v5 = *(unsigned int *)(a1 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a1 + 8 * v5) = -a2;
    v6 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
    *(_DWORD *)(a1 + 8) = v6;
    if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v6 )
    {
      sub_16CD150(a1, a1 + 16, 0, 8);
      v6 = *(unsigned int *)(a1 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a1 + 8 * v6) = 28;
    ++*(_DWORD *)(a1 + 8);
  }
}
