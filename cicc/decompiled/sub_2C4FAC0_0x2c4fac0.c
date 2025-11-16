// Function: sub_2C4FAC0
// Address: 0x2c4fac0
//
_BYTE *__fastcall sub_2C4FAC0(__int64 *a1, _BYTE *a2, int a3)
{
  _BYTE *result; // rax
  _BYTE *v4; // r12
  __int64 v5; // rdi
  _QWORD *v6; // rax
  _QWORD *v7; // rcx
  __int64 *v8; // rax
  int v9; // [rsp-1Ch] [rbp-1Ch]

  result = a2;
  if ( *a2 == 92 )
  {
    if ( (unsigned int)**((unsigned __int8 **)a2 - 4) - 12 > 1 )
      return *(_BYTE **)&a2[32 * a3 - 64];
    v4 = (_BYTE *)*((_QWORD *)a2 - 8);
    if ( *v4 != 92 )
      return *(_BYTE **)&a2[32 * a3 - 64];
    v5 = *a1;
    if ( *(_BYTE *)(v5 + 28) )
    {
      v6 = *(_QWORD **)(v5 + 8);
      v7 = &v6[*(unsigned int *)(v5 + 20)];
      if ( v6 == v7 )
        return *(_BYTE **)&a2[32 * a3 - 64];
      while ( v4 != (_BYTE *)*v6 )
      {
        if ( v7 == ++v6 )
          return *(_BYTE **)&a2[32 * a3 - 64];
      }
    }
    else
    {
      v9 = a3;
      v8 = sub_C8CA60(v5, *((_QWORD *)a2 - 8));
      a3 = v9;
      if ( !v8 )
        return *(_BYTE **)&a2[32 * a3 - 64];
    }
    return *(_BYTE **)&v4[32 * a3 - 64];
  }
  return result;
}
