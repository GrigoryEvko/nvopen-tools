// Function: sub_26E65E0
// Address: 0x26e65e0
//
_QWORD *__fastcall sub_26E65E0(_QWORD *a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  _QWORD *v4; // r15
  _QWORD *v5; // rbx
  unsigned __int64 v10; // rcx
  _QWORD *v11; // r8
  size_t v12; // rdx
  const void *v13; // rdi
  const void *v14; // rsi
  int v15; // eax
  _QWORD *v17; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD **)(*a1 + 8 * a2);
  if ( !v4 )
    return 0;
  v5 = (_QWORD *)*v4;
  v10 = *(_QWORD *)(*v4 + 40LL);
  while ( 1 )
  {
    if ( v10 == a4 && *a3 == v5[1] )
    {
      v12 = a3[2];
      if ( v12 == v5[3] )
      {
        v13 = (const void *)a3[1];
        v14 = (const void *)v5[2];
        if ( v13 == v14 )
          break;
        if ( v13 )
        {
          if ( v14 )
          {
            v17 = a3;
            v15 = memcmp(v13, v14, v12);
            a3 = v17;
            if ( !v15 )
              break;
          }
        }
      }
    }
    v11 = (_QWORD *)*v5;
    if ( !*v5 )
      return v11;
    v10 = v11[5];
    v4 = v5;
    if ( a2 != v10 % a1[1] )
      return 0;
    v5 = (_QWORD *)*v5;
  }
  return v4;
}
