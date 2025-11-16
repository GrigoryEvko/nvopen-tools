// Function: sub_1B751F0
// Address: 0x1b751f0
//
__int64 __fastcall sub_1B751F0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  _QWORD *v9; // rax
  __int64 result; // rax

  v7 = *a1;
  v8 = *(unsigned int *)(*a1 + 32);
  if ( (unsigned int)v8 >= *(_DWORD *)(*a1 + 36) )
  {
    sub_16CD150(v7 + 24, (const void *)(v7 + 40), 0, 16, a5, a6);
    v8 = *(unsigned int *)(v7 + 32);
  }
  v9 = (_QWORD *)(*(_QWORD *)(v7 + 24) + 16 * v8);
  *v9 = a2;
  v9[1] = a3;
  result = *(unsigned int *)(v7 + 32);
  *(_DWORD *)(v7 + 32) = result + 1;
  return result;
}
