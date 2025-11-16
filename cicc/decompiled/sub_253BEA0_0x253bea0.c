// Function: sub_253BEA0
// Address: 0x253bea0
//
_QWORD *__fastcall sub_253BEA0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rdx
  _QWORD *v3; // rcx
  _QWORD *result; // rax
  __int64 v5; // r8
  __int64 v6; // rdi
  int v7; // esi
  _QWORD *v8; // rdx
  char v9; // r8

  v2 = *(unsigned int *)(a2 + 248);
  v3 = *(_QWORD **)(a2 + 232);
  result = a1;
  v5 = a2 + 224;
  v6 = *(_QWORD *)(a2 + 224);
  v7 = *(_DWORD *)(a2 + 240);
  *result = v5;
  result[1] = v6;
  v8 = &v3[12 * v2];
  if ( !v7 )
  {
    result[2] = v8;
    result[3] = v8;
    return result;
  }
  result[2] = v3;
  result[3] = v8;
  if ( v3 != v8 )
  {
    v9 = 0;
    while ( 1 )
    {
      if ( *v3 == 0x7FFFFFFFFFFFFFFFLL )
      {
        if ( v3[1] != 0x7FFFFFFFFFFFFFFFLL )
          goto LABEL_7;
      }
      else if ( *v3 != 0x7FFFFFFFFFFFFFFELL || v3[1] != 0x7FFFFFFFFFFFFFFELL )
      {
LABEL_7:
        if ( !v9 )
          return result;
LABEL_11:
        result[2] = v3;
        return result;
      }
      v3 += 12;
      v9 = 1;
      if ( v3 == v8 )
        goto LABEL_11;
    }
  }
  return result;
}
