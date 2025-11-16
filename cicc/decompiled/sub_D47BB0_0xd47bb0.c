// Function: sub_D47BB0
// Address: 0xd47bb0
//
__int64 __fastcall sub_D47BB0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rbx
  __int64 *v4; // r13
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 result; // rax
  unsigned int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // rdi

  v3 = *(__int64 **)(a1 + 8);
  v4 = *(__int64 **)(a1 + 16);
  if ( v3 == v4 )
  {
    *(_BYTE *)(a1 + 152) = 1;
  }
  else
  {
    do
    {
      v5 = *v3++;
      sub_D47BB0(v5);
    }
    while ( v4 != v3 );
    *(_BYTE *)(a1 + 152) = 1;
    v6 = *(_QWORD *)(a1 + 8);
    if ( v6 != *(_QWORD *)(a1 + 16) )
      *(_QWORD *)(a1 + 16) = v6;
  }
  result = *(_QWORD *)(a1 + 32);
  if ( result != *(_QWORD *)(a1 + 40) )
    *(_QWORD *)(a1 + 40) = result;
  ++*(_QWORD *)(a1 + 56);
  if ( *(_BYTE *)(a1 + 84) )
  {
    *(_QWORD *)a1 = 0;
  }
  else
  {
    v8 = 4 * (*(_DWORD *)(a1 + 76) - *(_DWORD *)(a1 + 80));
    v9 = *(unsigned int *)(a1 + 72);
    if ( v8 < 0x20 )
      v8 = 32;
    if ( (unsigned int)v9 > v8 )
    {
      sub_C8C990(a1 + 56, a2);
    }
    else
    {
      a2 = 0xFFFFFFFFLL;
      memset(*(void **)(a1 + 64), -1, 8 * v9);
    }
    result = *(unsigned __int8 *)(a1 + 84);
    *(_QWORD *)a1 = 0;
    if ( !(_BYTE)result )
      result = _libc_free(*(_QWORD *)(a1 + 64), a2);
  }
  v10 = *(_QWORD *)(a1 + 32);
  if ( v10 )
    result = j_j___libc_free_0(v10, *(_QWORD *)(a1 + 48) - v10);
  v11 = *(_QWORD *)(a1 + 8);
  if ( v11 )
    return j_j___libc_free_0(v11, *(_QWORD *)(a1 + 24) - v11);
  return result;
}
