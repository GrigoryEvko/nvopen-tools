// Function: sub_1636020
// Address: 0x1636020
//
void __fastcall sub_1636020(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  __int64 v4; // rax

  if ( *(_BYTE *)(a2 + 40) )
  {
    v2 = *(_QWORD *)(a1 + 8);
    v3 = *(_QWORD *)(a2 + 32);
    v4 = *(unsigned int *)(v2 + 8);
    if ( (unsigned int)v4 >= *(_DWORD *)(v2 + 12) )
    {
      sub_16CD150(*(_QWORD *)(a1 + 8), v2 + 16, 0, 8);
      v4 = *(unsigned int *)(v2 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v2 + 8 * v4) = v3;
    ++*(_DWORD *)(v2 + 8);
  }
}
