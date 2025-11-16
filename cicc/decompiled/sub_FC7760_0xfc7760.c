// Function: sub_FC7760
// Address: 0xfc7760
//
__int64 __fastcall sub_FC7760(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  _QWORD *v9; // rax
  __int64 result; // rax

  v7 = *a1;
  v8 = *(unsigned int *)(*a1 + 32);
  if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(*a1 + 36) )
  {
    sub_C8D5F0(v7 + 24, (const void *)(v7 + 40), v8 + 1, 0x10u, a5, a6);
    v8 = *(unsigned int *)(v7 + 32);
  }
  v9 = (_QWORD *)(*(_QWORD *)(v7 + 24) + 16 * v8);
  *v9 = a2;
  v9[1] = a3;
  result = *(unsigned int *)(v7 + 32);
  *(_DWORD *)(v7 + 32) = result + 1;
  return result;
}
