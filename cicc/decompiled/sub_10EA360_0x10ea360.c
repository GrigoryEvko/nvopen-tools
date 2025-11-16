// Function: sub_10EA360
// Address: 0x10ea360
//
unsigned __int8 *__fastcall sub_10EA360(__int64 *a1, __int64 a2)
{
  unsigned __int8 *v2; // r13
  unsigned __int8 *result; // rax
  __int64 v4; // rcx
  unsigned __int8 *v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // r13
  __int64 v9; // r13
  __int64 v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(unsigned __int8 **)a2;
  result = sub_98ACB0(*(unsigned __int8 **)a2, 0);
  if ( result != v2 )
  {
    v4 = *a1;
    v5 = *(unsigned __int8 **)a2;
    if ( *(_QWORD *)a2 )
    {
      v6 = *(_QWORD *)(a2 + 8);
      **(_QWORD **)(a2 + 16) = v6;
      if ( v6 )
        *(_QWORD *)(v6 + 16) = *(_QWORD *)(a2 + 16);
    }
    *(_QWORD *)a2 = result;
    if ( result )
    {
      v7 = *((_QWORD *)result + 2);
      *(_QWORD *)(a2 + 8) = v7;
      if ( v7 )
        *(_QWORD *)(v7 + 16) = a2 + 8;
      *(_QWORD *)(a2 + 16) = result + 16;
      *((_QWORD *)result + 2) = a2;
    }
    if ( *v5 > 0x1Cu )
    {
      v8 = *(_QWORD *)(v4 + 40);
      v10[0] = (__int64)v5;
      v9 = v8 + 2096;
      sub_10E8740(v9, v10);
      result = (unsigned __int8 *)*((_QWORD *)v5 + 2);
      if ( result )
      {
        if ( !*((_QWORD *)result + 1) )
        {
          v10[0] = *((_QWORD *)result + 3);
          return (unsigned __int8 *)sub_10E8740(v9, v10);
        }
      }
    }
  }
  return result;
}
