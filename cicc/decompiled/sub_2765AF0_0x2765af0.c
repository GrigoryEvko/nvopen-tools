// Function: sub_2765AF0
// Address: 0x2765af0
//
void __fastcall sub_2765AF0(__int64 *a1, __int64 *a2)
{
  __int64 *v3; // r12
  __int64 v5; // rdi
  __int64 v6; // rsi
  __int64 *v7; // r13
  __int64 v8; // rsi
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rdx

  if ( a1 != a2 )
  {
    v3 = a1 + 2;
    if ( a2 != a1 + 2 )
    {
      v5 = *v3;
      v6 = *a1;
      if ( *v3 == *a1 )
        goto LABEL_9;
LABEL_4:
      v7 = v3 + 2;
      if ( sub_B445A0(v5, v6) )
      {
LABEL_5:
        v8 = *(v7 - 2);
        v9 = *(v7 - 1);
        v10 = ((char *)v3 - (char *)a1) >> 4;
        if ( (char *)v3 - (char *)a1 > 0 )
        {
          do
          {
            v11 = *(v3 - 2);
            v3 -= 2;
            v3[2] = v11;
            v3[3] = v3[1];
            --v10;
          }
          while ( v10 );
        }
        *a1 = v8;
        a1[1] = v9;
        if ( a2 != v7 )
          goto LABEL_8;
      }
      else
      {
        while ( 1 )
        {
          sub_2765A80(v3);
          if ( a2 == v7 )
            break;
LABEL_8:
          v3 = v7;
          v6 = *a1;
          v5 = *v7;
          if ( *v7 != *a1 )
            goto LABEL_4;
LABEL_9:
          v7 = v3 + 2;
          if ( sub_B445A0(v3[1], a1[1]) )
            goto LABEL_5;
        }
      }
    }
  }
}
