// Function: sub_2ABD9E0
// Address: 0x2abd9e0
//
_QWORD *__fastcall sub_2ABD9E0(_QWORD *a1, __int64 *a2)
{
  _QWORD *v2; // rdx
  int v4; // ecx
  _QWORD *v5; // rdi
  __int64 v6; // rax
  bool v8; // al
  bool v9; // si

  v2 = (_QWORD *)a2[1];
  v4 = *((_DWORD *)a2 + 4);
  v5 = &v2[3 * *((unsigned int *)a2 + 6)];
  v6 = *a2;
  *a1 = a2;
  a1[1] = v6;
  if ( v4 )
  {
    a1[2] = v2;
    a1[3] = v5;
    if ( v2 != v5 )
    {
      v8 = 0;
      while ( 1 )
      {
        v9 = v8;
        v8 = *v2 == -8192 || *v2 == -4096;
        if ( !v8 )
          break;
        v2 += 3;
        if ( v2 == v5 )
          goto LABEL_8;
      }
      if ( v9 )
      {
LABEL_8:
        a1[2] = v2;
        return a1;
      }
    }
  }
  else
  {
    a1[2] = v5;
    a1[3] = v5;
  }
  return a1;
}
