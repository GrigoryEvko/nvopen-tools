// Function: sub_1E058E0
// Address: 0x1e058e0
//
_QWORD *__fastcall sub_1E058E0(_QWORD *a1, __int64 *a2)
{
  __int64 v2; // rcx
  _QWORD *v3; // rdx
  int v5; // edi
  __int64 v6; // rax
  _QWORD *v7; // rcx
  bool v9; // al
  bool v10; // di

  v2 = *((unsigned int *)a2 + 6);
  v3 = (_QWORD *)a2[1];
  v5 = *((_DWORD *)a2 + 4);
  v6 = *a2;
  *a1 = a2;
  v7 = &v3[2 * v2];
  a1[1] = v6;
  if ( v5 )
  {
    a1[2] = v3;
    a1[3] = v7;
    if ( v3 != v7 )
    {
      v9 = 0;
      while ( 1 )
      {
        v10 = v9;
        v9 = *v3 == -16 || *v3 == -8;
        if ( !v9 )
          break;
        v3 += 2;
        if ( v3 == v7 )
          goto LABEL_8;
      }
      if ( v10 )
      {
LABEL_8:
        a1[2] = v3;
        return a1;
      }
    }
  }
  else
  {
    a1[2] = v7;
    a1[3] = v7;
  }
  return a1;
}
