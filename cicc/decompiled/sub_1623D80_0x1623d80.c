// Function: sub_1623D80
// Address: 0x1623d80
//
__int64 *__fastcall sub_1623D80(
        __int64 a1,
        __int64 a2,
        char a3,
        char a4,
        __int64 *a5,
        __int64 a6,
        __int64 *a7,
        __int64 *a8)
{
  __int64 *v8; // r14
  __int64 *v9; // r13
  __int64 *v10; // r12
  unsigned int v11; // ebx
  __int64 v12; // rdx
  unsigned int v13; // esi
  __int64 *result; // rax
  __int64 *v15; // r12
  __int64 v16; // rdx
  unsigned int v17; // esi

  v8 = &a5[a6];
  *(_BYTE *)a1 = a3;
  *(_BYTE *)(a1 + 1) = a4;
  v9 = a7;
  *(_WORD *)(a1 + 2) = 0;
  *(_DWORD *)(a1 + 4) = 0;
  *(_QWORD *)(a1 + 8) = (unsigned int)(a6 + (_DWORD)a8);
  *(_QWORD *)(a1 + 16) = a2;
  if ( a5 == v8 )
  {
    v11 = 0;
  }
  else
  {
    v10 = a5;
    v11 = 0;
    do
    {
      v12 = *v10;
      v13 = v11;
      ++v10;
      ++v11;
      sub_1623D00(a1, v13, v12);
    }
    while ( v8 != v10 );
  }
  result = a8;
  v15 = &a7[(_QWORD)a8];
  if ( a7 != v15 )
  {
    do
    {
      v16 = *v9;
      v17 = v11;
      ++v9;
      ++v11;
      result = (__int64 *)sub_1623D00(a1, v17, v16);
    }
    while ( v15 != v9 );
  }
  if ( !*(_BYTE *)(a1 + 1) )
    return sub_161EA20(a1);
  return result;
}
