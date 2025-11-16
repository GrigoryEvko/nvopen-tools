// Function: sub_25DCD90
// Address: 0x25dcd90
//
__int64 __fastcall sub_25DCD90(__int64 a1, _BYTE *a2, __int64 a3)
{
  _BYTE *v4; // rbx
  char v5; // al
  __int64 result; // rax
  __int64 v7; // r9
  __int64 v8; // rdx
  __int64 v9; // r13
  __int64 v10; // r14
  unsigned __int8 v11; // [rsp+8h] [rbp-28h]

  v4 = a2;
  v5 = *a2;
  if ( *a2 )
  {
    while ( v5 == 86 )
    {
      if ( !(unsigned __int8)sub_25DCD90(a1, *((_QWORD *)v4 - 8), a3) )
        return 0;
      v4 = (_BYTE *)*((_QWORD *)v4 - 4);
      v5 = *v4;
      if ( !*v4 )
        goto LABEL_2;
    }
    if ( v5 != 84 )
      return 0;
    v9 = 0;
    v10 = 32LL * (*((_DWORD *)v4 + 1) & 0x7FFFFFF);
    if ( (*((_DWORD *)v4 + 1) & 0x7FFFFFF) != 0 )
    {
      while ( (unsigned __int8)sub_25DCD90(a1, *(_QWORD *)(*((_QWORD *)v4 - 1) + v9), a3) )
      {
        v9 += 32;
        if ( v10 == v9 )
          return 1;
      }
      return 0;
    }
    return 1;
  }
  else
  {
LABEL_2:
    result = sub_DFE490(a1);
    if ( !(_BYTE)result )
      return 0;
    v8 = *(unsigned int *)(a3 + 8);
    if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
      v11 = result;
      sub_C8D5F0(a3, (const void *)(a3 + 16), v8 + 1, 8u, v8 + 1, v7);
      v8 = *(unsigned int *)(a3 + 8);
      result = v11;
    }
    *(_QWORD *)(*(_QWORD *)a3 + 8 * v8) = v4;
    ++*(_DWORD *)(a3 + 8);
  }
  return result;
}
