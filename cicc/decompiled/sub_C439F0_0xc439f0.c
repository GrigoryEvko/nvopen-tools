// Function: sub_C439F0
// Address: 0xc439f0
//
__int64 __fastcall sub_C439F0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  int v4; // r12d
  __int64 result; // rax
  unsigned int v6; // edx
  __int64 v7; // r13
  __int64 v8; // rsi
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // r12
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // rcx
  __int64 v13; // rax
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // r12
  unsigned __int64 v16; // rcx

  v3 = *(unsigned int *)(a2 + 8);
  v4 = *((_DWORD *)a1 + 2);
  if ( v3 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, a2 + 16, v3 + 1, 4);
    v3 = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * v3) = v4;
  result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
  *(_DWORD *)(a2 + 8) = result;
  v6 = *((_DWORD *)a1 + 2);
  if ( v6 <= 0x40 )
  {
    v14 = *a1;
    if ( result + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      sub_C8D5F0(a2, a2 + 16, result + 1, 4);
      result = *(unsigned int *)(a2 + 8);
    }
    *(_DWORD *)(*(_QWORD *)a2 + 4 * result) = v14;
    v15 = HIDWORD(v14);
    v16 = *(unsigned int *)(a2 + 12);
    result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = result;
    if ( result + 1 > v16 )
    {
      sub_C8D5F0(a2, a2 + 16, result + 1, 4);
      result = *(unsigned int *)(a2 + 8);
    }
    *(_DWORD *)(*(_QWORD *)a2 + 4 * result) = v15;
    ++*(_DWORD *)(a2 + 8);
  }
  else
  {
    v7 = 0;
    v8 = a2 + 16;
    v9 = ((unsigned __int64)v6 + 63) >> 6;
    do
    {
      v10 = *(_QWORD *)(*a1 + 8 * v7);
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        sub_C8D5F0(a2, v8, result + 1, 4);
        result = *(unsigned int *)(a2 + 8);
      }
      *(_DWORD *)(*(_QWORD *)a2 + 4 * result) = v10;
      v11 = HIDWORD(v10);
      v12 = *(unsigned int *)(a2 + 12);
      v13 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = v13;
      if ( v13 + 1 > v12 )
      {
        sub_C8D5F0(a2, v8, v13 + 1, 4);
        v13 = *(unsigned int *)(a2 + 8);
      }
      ++v7;
      *(_DWORD *)(*(_QWORD *)a2 + 4 * v13) = v11;
      result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = result;
    }
    while ( (unsigned int)v9 > (unsigned int)v7 );
  }
  return result;
}
