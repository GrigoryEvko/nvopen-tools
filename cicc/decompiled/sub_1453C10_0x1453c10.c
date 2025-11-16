// Function: sub_1453C10
// Address: 0x1453c10
//
void __fastcall sub_1453C10(__int64 a1, __int64 a2)
{
  __int64 i; // r12
  __int64 v3; // r14
  __int64 v4; // rax

  for ( i = *(_QWORD *)(a1 + 8); i; i = *(_QWORD *)(i + 8) )
  {
    v3 = sub_1648700(i);
    v4 = *(unsigned int *)(a2 + 8);
    if ( (unsigned int)v4 >= *(_DWORD *)(a2 + 12) )
    {
      sub_16CD150(a2, a2 + 16, 0, 8);
      v4 = *(unsigned int *)(a2 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a2 + 8 * v4) = v3;
    ++*(_DWORD *)(a2 + 8);
  }
}
