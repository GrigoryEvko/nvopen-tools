// Function: sub_143CDC0
// Address: 0x143cdc0
//
__int64 __fastcall sub_143CDC0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v7; // r13d
  __int64 v11; // rsi
  __int64 result; // rax
  __int64 v13; // r12
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 v16; // rdx

  v7 = *(_DWORD *)(a5 + 8);
  v11 = *a1;
  result = sub_143C5C0((__int64)a1, *a1, a2, a3, a4, a5);
  *a1 = result;
  v13 = result;
  if ( !result )
  {
    while ( 1 )
    {
      v16 = *(unsigned int *)(a5 + 8);
      if ( v7 == (_DWORD)v16 )
        break;
      v14 = *(_QWORD *)a5;
      v15 = *(_QWORD *)(*(_QWORD *)a5 + 8 * v16 - 8);
      *(_DWORD *)(a5 + 8) = v16 - 1;
      sub_15F20C0(v15, v11, v16, v14);
    }
    return v13;
  }
  return result;
}
