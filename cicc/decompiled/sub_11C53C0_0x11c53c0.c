// Function: sub_11C53C0
// Address: 0x11c53c0
//
__int64 __fastcall sub_11C53C0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  size_t v8; // r12
  __int64 v9; // rdi
  const void *v10; // r15
  __int64 v11; // r12
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 result; // rax
  __int64 v15; // rdx

  v8 = *(_QWORD *)(a2 + 8);
  v9 = a3[1];
  v10 = *(const void **)a2;
  if ( v8 + v9 > a3[2] )
  {
    sub_C8D290((__int64)a3, a3 + 3, v8 + v9, 1u, a5, a6);
    v9 = a3[1];
  }
  if ( v8 )
  {
    memcpy((void *)(*a3 + v9), v10, v8);
    v9 = a3[1];
  }
  v11 = v9 + v8;
  v12 = a3[2];
  a3[1] = v11;
  v13 = v11 + 1;
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) == 2 )
  {
    if ( v13 > v12 )
    {
      sub_C8D290((__int64)a3, a3 + 3, v13, 1u, a5, a6);
      v11 = a3[1];
    }
    *(_BYTE *)(*a3 + v11) = 102;
    result = a3[1] + 1;
    a3[1] = result;
  }
  else
  {
    if ( v13 > v12 )
    {
      sub_C8D290((__int64)a3, a3 + 3, v13, 1u, a5, a6);
      v11 = a3[1];
    }
    *(_BYTE *)(*a3 + v11) = 108;
    result = a3[1] + 1;
    a3[1] = result;
  }
  v15 = *a3;
  *(_QWORD *)(a2 + 8) = result;
  *(_QWORD *)a2 = v15;
  return result;
}
