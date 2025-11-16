// Function: sub_1D5BD90
// Address: 0x1d5bd90
//
__int64 __fastcall sub_1D5BD90(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v5; // rbx
  __int64 v6; // rax
  _QWORD *v7; // r12
  _QWORD *v8; // rax
  __int64 v9; // rcx
  unsigned __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 result; // rax
  _QWORD *v13; // rdx

  v5 = a3;
  v6 = sub_22077B0(32);
  v7 = (_QWORD *)v6;
  if ( v6 )
  {
    *(_QWORD *)(v6 + 8) = a2;
    *(_QWORD *)v6 = off_4985558;
    *(_DWORD *)(v6 + 24) = v5;
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v8 = (_QWORD *)(*(_QWORD *)(a2 - 8) + 24 * v5);
    else
      v8 = (_QWORD *)(a2 + 24 * (v5 - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    v7[2] = *v8;
    if ( *v8 )
    {
      v9 = v8[1];
      v10 = v8[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v10 = v9;
      if ( v9 )
      {
        a2 = *(_QWORD *)(v9 + 16) & 3LL;
        *(_QWORD *)(v9 + 16) = a2 | v10;
      }
    }
    *v8 = a4;
    if ( a4 )
    {
      v11 = *(_QWORD *)(a4 + 8);
      a2 = a4 + 8;
      v8[1] = v11;
      if ( v11 )
        *(_QWORD *)(v11 + 16) = (unsigned __int64)(v8 + 1) | *(_QWORD *)(v11 + 16) & 3LL;
      v8[2] = a2 | v8[2] & 3LL;
      *(_QWORD *)(a4 + 8) = v8;
    }
  }
  result = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)result >= *(_DWORD *)(a1 + 12) )
  {
    sub_1D5B850(a1, a2);
    result = *(unsigned int *)(a1 + 8);
  }
  v13 = (_QWORD *)(*(_QWORD *)a1 + 8LL * (unsigned int)result);
  if ( v13 )
  {
    *v13 = v7;
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    result = (unsigned int)(result + 1);
    *(_DWORD *)(a1 + 8) = result;
    if ( v7 )
      return (*(__int64 (__fastcall **)(_QWORD *))(*v7 + 8LL))(v7);
  }
  return result;
}
