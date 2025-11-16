// Function: sub_1040630
// Address: 0x1040630
//
_QWORD *__fastcall sub_1040630(_QWORD **a1, __int64 a2, __int64 a3, _BYTE *a4)
{
  _QWORD *result; // rax
  _QWORD *v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rcx

  result = sub_1040050((__int64)a1, a2, *a1, a4);
  if ( result )
  {
    v6 = result - 8;
    if ( *(_BYTE *)result == 26 )
      v6 = result - 4;
    if ( *v6 )
    {
      v7 = v6[1];
      *(_QWORD *)v6[2] = v7;
      if ( v7 )
        *(_QWORD *)(v7 + 16) = v6[2];
    }
    *v6 = a3;
    if ( a3 )
    {
      v8 = *(_QWORD *)(a3 + 16);
      v6[1] = v8;
      if ( v8 )
        *(_QWORD *)(v8 + 16) = v6 + 1;
      v6[2] = a3 + 16;
      *(_QWORD *)(a3 + 16) = v6;
    }
  }
  return result;
}
