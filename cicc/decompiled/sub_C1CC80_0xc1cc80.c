// Function: sub_C1CC80
// Address: 0xc1cc80
//
_QWORD *__fastcall sub_C1CC80(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // r13
  _QWORD *v5; // rbx
  unsigned __int64 v10; // rcx
  size_t v11; // rdx
  const void *v12; // rdi
  const void *v13; // rsi
  int v14; // eax
  __int64 v16; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD **)(*a1 + 8 * a2);
  if ( v4 )
  {
    v5 = (_QWORD *)*v4;
    v10 = *(_QWORD *)(*v4 + 32LL);
    while ( 1 )
    {
      if ( v10 == a4 )
      {
        v11 = *(_QWORD *)(a3 + 8);
        if ( v11 == v5[2] )
        {
          v12 = *(const void **)a3;
          v13 = (const void *)v5[1];
          if ( *(const void **)a3 == v13 )
            break;
          if ( v12 )
          {
            if ( v13 )
            {
              v16 = a3;
              v14 = memcmp(v12, v13, v11);
              a3 = v16;
              if ( !v14 )
                break;
            }
          }
        }
      }
      if ( !*v5 )
        return 0;
      v10 = *(_QWORD *)(*v5 + 32LL);
      v4 = v5;
      if ( a2 != v10 % a1[1] )
        return 0;
      v5 = (_QWORD *)*v5;
    }
  }
  return v4;
}
