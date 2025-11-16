// Function: sub_AF6280
// Address: 0xaf6280
//
void __fastcall sub_AF6280(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rcx
  __int64 v4; // rax
  __int64 v5; // rax

  if ( a2 > 0 )
  {
    v2 = *(unsigned int *)(a1 + 8);
    if ( v2 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      sub_C8D5F0(a1, a1 + 16, v2 + 1, 8);
      v2 = *(unsigned int *)(a1 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a1 + 8 * v2) = 35;
    v3 = *(unsigned int *)(a1 + 12);
    v4 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
    *(_DWORD *)(a1 + 8) = v4;
    if ( v4 + 1 > v3 )
    {
      sub_C8D5F0(a1, a1 + 16, v4 + 1, 8);
      v4 = *(unsigned int *)(a1 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a1 + 8 * v4) = a2;
    ++*(_DWORD *)(a1 + 8);
  }
  else if ( a2 )
  {
    v5 = *(unsigned int *)(a1 + 8);
    if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      sub_C8D5F0(a1, a1 + 16, v5 + 1, 8);
      v5 = *(unsigned int *)(a1 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a1 + 8 * v5) = 16;
    ++*(_DWORD *)(a1 + 8);
    sub_A188E0(a1, -a2);
    sub_A188E0(a1, 28);
  }
}
