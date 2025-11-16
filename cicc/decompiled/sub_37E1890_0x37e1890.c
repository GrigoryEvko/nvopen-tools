// Function: sub_37E1890
// Address: 0x37e1890
//
__int64 __fastcall sub_37E1890(
        __int64 a1,
        __int64 *a2,
        unsigned __int64 a3,
        unsigned __int8 (__fastcall *a4)(__int64, __int64))
{
  __int64 result; // rax
  __int64 *v7; // rbx
  __int64 v8; // r14
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // [rsp+8h] [rbp-38h]

  result = (__int64)a2 - a1;
  v7 = a2;
  if ( (__int64)a2 - a1 > 16 )
  {
    v11 = result >> 4;
    v8 = ((result >> 4) - 2) / 2;
    result = sub_37E1680(a1, v8, result >> 4, *(_QWORD *)(a1 + 16 * v8), *(_QWORD *)(a1 + 16 * v8 + 8), a4);
    while ( v8 )
    {
      --v8;
      result = sub_37E1680(a1, v8, v11, *(_QWORD *)(a1 + 16 * v8), *(_QWORD *)(a1 + 16 * v8 + 8), a4);
    }
  }
  if ( (unsigned __int64)a2 < a3 )
  {
    do
    {
      while ( 1 )
      {
        result = ((__int64 (__fastcall *)(__int64 *, __int64))a4)(v7, a1);
        if ( (_BYTE)result )
          break;
        v7 += 2;
        if ( a3 <= (unsigned __int64)v7 )
          return result;
      }
      v9 = *v7;
      v10 = v7[1];
      v7 += 2;
      *(v7 - 2) = *(_QWORD *)a1;
      *((_DWORD *)v7 - 2) = *(_DWORD *)(a1 + 8);
      result = sub_37E1680(a1, 0, ((__int64)a2 - a1) >> 4, v9, v10, a4);
    }
    while ( a3 > (unsigned __int64)v7 );
  }
  return result;
}
