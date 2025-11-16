// Function: sub_1ED7D20
// Address: 0x1ed7d20
//
__int64 __fastcall sub_1ED7D20(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 result; // rax
  unsigned int *v5; // rax

  v3 = *(_QWORD *)(a1 + 112) + 40LL * a2;
  result = *(unsigned __int8 *)(v3 + 33);
  if ( !*(_WORD *)(v3 + 33) && (unsigned int)(*(_DWORD *)v3 - 1) <= 1 )
  {
    v5 = *(unsigned int **)(v3 + 24);
    *(_BYTE *)(v3 + 34) = 1;
    result = sub_1ED7D20(a3, *v5, a1);
    *(_BYTE *)(v3 + 33) = result;
  }
  return result;
}
