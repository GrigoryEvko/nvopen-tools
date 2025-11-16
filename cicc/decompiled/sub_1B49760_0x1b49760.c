// Function: sub_1B49760
// Address: 0x1b49760
//
__int64 __fastcall sub_1B49760(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 result; // rax
  __int64 v8; // rbx
  __int64 v9; // r15
  unsigned __int64 v10; // r12
  _QWORD *v11; // r15
  __int64 v12; // [rsp+8h] [rbp-38h]

  result = *(unsigned int *)(a1 + 8);
  v12 = result;
  if ( a3 == a2 )
  {
    LODWORD(v10) = 0;
  }
  else
  {
    v8 = a2;
    v9 = a2;
    v10 = 0;
    do
    {
      do
        v9 = *(_QWORD *)(v9 + 8);
      while ( v9 && (unsigned __int8)(*((_BYTE *)sub_1648700(v9) + 16) - 25) > 9u );
      ++v10;
    }
    while ( a3 != v9 );
    if ( (unsigned __int64)*(unsigned int *)(a1 + 12) - v12 < v10 )
    {
      sub_16CD150(a1, (const void *)(a1 + 16), v10 + v12, 8, a5, a6);
      v12 = *(unsigned int *)(a1 + 8);
    }
    v11 = (_QWORD *)(*(_QWORD *)a1 + 8 * v12);
    do
    {
      if ( v11 )
        *v11 = sub_1648700(v8)[5];
      do
        v8 = *(_QWORD *)(v8 + 8);
      while ( v8 && (unsigned __int8)(*((_BYTE *)sub_1648700(v8) + 16) - 25) > 9u );
      ++v11;
    }
    while ( a3 != v8 );
    result = *(unsigned int *)(a1 + 8);
    LODWORD(v12) = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = v12 + v10;
  return result;
}
