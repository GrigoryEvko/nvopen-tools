// Function: sub_86C360
// Address: 0x86c360
//
void __fastcall sub_86C360(__int64 a1, __int64 a2)
{
  __int64 v4; // r10
  __int64 v5; // rdi
  __int64 v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // rax
  _BYTE *v9; // r12
  const __m128i *v10; // r13
  __m128i *v11; // rdx
  __int64 i; // rax
  __int64 v13; // rax
  __int64 j; // rax
  __m128i *v15[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_QWORD *)(a1 + 16);
  if ( v4 )
  {
    v5 = *(_QWORD *)(a2 + 16);
    v6 = v4;
    v7 = 0;
    while ( 1 )
    {
      if ( qword_4F5FD68 == a1 )
        v7 = 0;
      if ( v5 )
      {
        if ( v5 == v6 )
        {
          if ( qword_4F5FD68 == a1 )
            goto LABEL_30;
LABEL_21:
          if ( v4 == v6 )
            return;
LABEL_16:
          v9 = sub_86B560(a1, v7);
          if ( !v9 )
            return;
          v10 = *(const __m128i **)(a1 + 40);
          sub_732CD0(v10, v15);
          v11 = v15[0];
          *(__m128i **)(a1 + 40) = v15[0];
          v10[4].m128i_i64[1] = (__int64)v9;
          for ( i = *((_QWORD *)v9 + 2); i; i = *(_QWORD *)(i + 16) )
          {
            *((_QWORD *)v9 + 3) = v10;
            v9 = (_BYTE *)i;
          }
LABEL_19:
          *((_QWORD *)v9 + 3) = v10;
          *((_QWORD *)v9 + 2) = v11;
          return;
        }
        v8 = v5;
        while ( 1 )
        {
          if ( qword_4F5FD68 == a1 )
            v7 = v8;
          v8 = *(_QWORD *)(v8 + 16);
          if ( !v8 )
            break;
          if ( v8 == v6 )
          {
            if ( qword_4F5FD68 == a1 )
              goto LABEL_23;
            goto LABEL_21;
          }
        }
      }
      if ( qword_4F5FD68 != a1 )
        v7 = v6;
      v6 = *(_QWORD *)(v6 + 16);
      if ( !v6 )
      {
        if ( qword_4F5FD68 == a1 )
          goto LABEL_29;
        goto LABEL_16;
      }
    }
  }
  if ( qword_4F5FD68 == a1 )
  {
    v5 = *(_QWORD *)(a2 + 16);
    v7 = 0;
LABEL_29:
    if ( v5 )
    {
LABEL_23:
      v9 = sub_86B560(a1, *(_QWORD *)(v7 + 40));
      if ( v9 )
      {
        v10 = *(const __m128i **)(a1 + 40);
        sub_732CD0(v10, v15);
        v11 = v15[0];
        *(__m128i **)(a1 + 40) = v15[0];
        v10[4].m128i_i64[1] = (__int64)v9;
        v13 = *((_QWORD *)v9 + 2);
        if ( !v13 )
          goto LABEL_19;
        do
        {
          *((_QWORD *)v9 + 3) = v10;
          v9 = (_BYTE *)v13;
          v13 = *(_QWORD *)(v13 + 16);
        }
        while ( v13 );
        *((_QWORD *)v9 + 3) = v10;
        *((_QWORD *)v9 + 2) = v11;
      }
    }
    else
    {
LABEL_30:
      v9 = sub_86B560(a1, a2);
      if ( v9 )
      {
        v10 = *(const __m128i **)(a1 + 40);
        sub_732CD0(v10, v15);
        v11 = v15[0];
        *(__m128i **)(a1 + 40) = v15[0];
        v10[4].m128i_i64[1] = (__int64)v9;
        for ( j = *((_QWORD *)v9 + 2); j; j = *(_QWORD *)(j + 16) )
        {
          *((_QWORD *)v9 + 3) = v10;
          v9 = (_BYTE *)j;
        }
        goto LABEL_19;
      }
    }
  }
}
