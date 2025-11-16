// Function: sub_B32D80
// Address: 0xb32d80
//
__int64 __fastcall sub_B32D80(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // rdx
  __int64 v4; // r14
  __int64 v5; // r13
  _QWORD *v6; // rbx
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  _QWORD *v9; // r13
  _QWORD *v10; // rdx
  __int64 result; // rax

  v3 = 32 * a3;
  v4 = v3 >> 5;
  v5 = v3;
  v6 = a2;
  v7 = *(unsigned int *)(a1 + 8);
  v8 = v7 + (v3 >> 5);
  if ( v8 > *(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, a1 + 16, v8, 8);
    v7 = *(unsigned int *)(a1 + 8);
  }
  v9 = (_QWORD *)((char *)a2 + v5);
  v10 = (_QWORD *)(*(_QWORD *)a1 + 8 * v7);
  if ( v9 != a2 )
  {
    do
    {
      if ( v10 )
        *v10 = *v6;
      v6 += 4;
      ++v10;
    }
    while ( v9 != v6 );
    v7 = *(unsigned int *)(a1 + 8);
  }
  result = v4 + v7;
  *(_DWORD *)(a1 + 8) = result;
  return result;
}
