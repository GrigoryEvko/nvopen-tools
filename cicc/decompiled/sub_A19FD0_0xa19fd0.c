// Function: sub_A19FD0
// Address: 0xa19fd0
//
unsigned __int64 **__fastcall sub_A19FD0(__int64 a1, _QWORD *a2)
{
  unsigned __int64 **result; // rax
  unsigned __int64 *v3; // r12
  unsigned __int64 *v5; // r14
  unsigned __int64 *v6; // r12
  unsigned __int64 *v7; // r14
  unsigned __int64 *v8; // rsi
  unsigned __int64 *v9; // r12
  unsigned __int64 *v10; // r14
  unsigned __int64 *v11; // rsi
  unsigned __int64 *v12; // r12
  unsigned __int64 *v13; // r14
  unsigned __int64 *v14; // rsi
  unsigned __int64 *v15; // r12
  unsigned __int64 *i; // r13
  unsigned __int64 *v17; // rsi
  unsigned __int64 *v18; // rsi

  result = *(unsigned __int64 ***)(a1 + 80);
  if ( result )
  {
    v3 = *result;
    v5 = result[1];
    if ( *result == v5 )
      goto LABEL_3;
    do
    {
      v18 = v3++;
      sub_A19EB0(a2, v18);
    }
    while ( v5 != v3 );
    result = *(unsigned __int64 ***)(a1 + 80);
    if ( result )
    {
LABEL_3:
      v6 = result[3];
      v7 = result[4];
      if ( v6 == v7 )
        goto LABEL_19;
      do
      {
        v8 = v6;
        v6 += 2;
        sub_A19EB0(a2, v8);
      }
      while ( v7 != v6 );
      result = *(unsigned __int64 ***)(a1 + 80);
      if ( result )
      {
LABEL_19:
        v9 = result[6];
        v10 = result[7];
        if ( v9 == v10 )
          goto LABEL_9;
        do
        {
          v11 = v9;
          v9 += 2;
          sub_A19EB0(a2, v11);
        }
        while ( v10 != v9 );
        result = *(unsigned __int64 ***)(a1 + 80);
        if ( result )
        {
LABEL_9:
          v12 = result[9];
          v13 = result[10];
          if ( v12 == v13 )
            goto LABEL_12;
          do
          {
            v14 = v12;
            v12 += 5;
            sub_A19EB0(a2, v14);
          }
          while ( v13 != v12 );
          result = *(unsigned __int64 ***)(a1 + 80);
          if ( result )
          {
LABEL_12:
            v15 = result[12];
            for ( i = result[13]; i != v15; result = (unsigned __int64 **)sub_A19EB0(a2, v17) )
            {
              v17 = v15;
              v15 += 5;
            }
          }
        }
      }
    }
  }
  return result;
}
