// Function: sub_12D5520
// Address: 0x12d5520
//
void __fastcall sub_12D5520(unsigned __int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rcx
  unsigned __int64 *i; // rdx
  _QWORD *v6; // rax
  __int64 *v7; // rax
  unsigned __int64 *v8; // r8
  __int64 *v9; // r15
  unsigned __int64 *v10; // r12
  __int64 v11; // rbx
  __int64 v12; // rsi
  unsigned __int64 *v13; // r13
  __int64 v14; // rsi
  char **v15; // rcx
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rax
  char **v18; // rbx
  unsigned __int64 v19; // rsi
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // rdx
  __int64 v22; // rcx
  char **v23; // [rsp+8h] [rbp-68h]
  __int64 *v24; // [rsp+10h] [rbp-60h]
  char **v25; // [rsp+18h] [rbp-58h]
  unsigned __int64 v26; // [rsp+20h] [rbp-50h] BYREF
  __int64 *v27; // [rsp+28h] [rbp-48h]
  __int64 v28[8]; // [rsp+30h] [rbp-40h] BYREF

  v24 = (__int64 *)a1;
  v3 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1 != (unsigned __int64 *)v3 )
  {
    i = (unsigned __int64 *)a1[1];
    v6 = (_QWORD *)i[1];
    if ( a1 != v6 )
    {
      if ( a1 != i )
      {
        for ( i = (unsigned __int64 *)i[1]; ; i = (unsigned __int64 *)i[1] )
        {
          v7 = (__int64 *)v6[1];
          if ( v24 == v7 )
            break;
          v6 = (_QWORD *)v7[1];
          if ( v24 == v6 )
            break;
        }
      }
      v8 = (unsigned __int64 *)v24;
      v25 = (char **)&v26;
      v27 = (__int64 *)&v26;
      v26 = (unsigned __int64)&v26 + 4;
      if ( v24 != (__int64 *)i )
      {
        *(_QWORD *)((*i & 0xFFFFFFFFFFFFFFF8LL) + 8) = v24;
        v23 = (char **)*v8;
        *v8 = *v8 & 7 | *i & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v3 + 8) = &v26;
        *i = (unsigned __int64)&v26 | *i & 7;
        v27 = (__int64 *)i;
        v26 = v26 & 7 | v3;
      }
      sub_12D5520(v24, a2, a3);
      sub_12D5520(v25, a2, a3);
      v28[0] = a2;
      v28[1] = a3;
      if ( v25 != (char **)(v26 & 0xFFFFFFFFFFFFFFF8LL) )
      {
        v9 = v27;
        v10 = (unsigned __int64 *)v24[1];
        v23 = (char **)v27;
        if ( v24 == (__int64 *)v10 )
        {
LABEL_30:
          v18 = v23;
          if ( v23 != v25 )
          {
            v19 = v26;
            v20 = v26 & 7;
            *(_QWORD *)((*v9 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v25;
            v19 &= 0xFFFFFFFFFFFFFFF8LL;
            v21 = v20 | *v9 & 0xFFFFFFFFFFFFFFF8LL;
            v22 = *v24;
            *(_QWORD *)(v19 + 8) = v24;
            v26 = v21;
            v22 &= 0xFFFFFFFFFFFFFFF8LL;
            *v9 = v22 | *v9 & 7;
            *(_QWORD *)(v22 + 8) = v18;
            *v24 = v19 | *v24 & 7;
          }
        }
        else
        {
          while ( 1 )
          {
            while ( 1 )
            {
              v11 = (__int64)(v10 - 7);
              if ( !v10 )
                v11 = 0;
              v12 = (__int64)(v9 - 7);
              if ( !v9 )
                v12 = 0;
              if ( sub_12D5220(v28, v12, v11) )
                break;
              v10 = (unsigned __int64 *)v10[1];
              if ( v24 == (__int64 *)v10 )
                goto LABEL_30;
            }
            v13 = (unsigned __int64 *)v9[1];
            if ( v13 == (unsigned __int64 *)v25 )
            {
LABEL_33:
              v13 = (unsigned __int64 *)v25;
            }
            else
            {
              while ( 1 )
              {
                v14 = (__int64)(v13 - 7);
                if ( !v13 )
                  v14 = 0;
                if ( !sub_12D5220(v28, v14, v11) )
                  break;
                v13 = (unsigned __int64 *)v13[1];
                if ( v13 == (unsigned __int64 *)v25 )
                  goto LABEL_33;
              }
            }
            if ( v13 != v10 )
            {
              v15 = v23;
              if ( v13 != (unsigned __int64 *)v23 )
              {
                v16 = *v13 & 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)((*v9 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v13;
                *v13 = *v13 & 7 | *v9 & 0xFFFFFFFFFFFFFFF8LL;
                v17 = *v10;
                *(_QWORD *)(v16 + 8) = v10;
                v17 &= 0xFFFFFFFFFFFFFFF8LL;
                *v9 = v17 | *v9 & 7;
                *(_QWORD *)(v17 + 8) = v15;
                *v10 = v16 | *v10 & 7;
              }
            }
            if ( v13 == (unsigned __int64 *)v25 )
              break;
            v23 = (char **)v13;
            v10 = (unsigned __int64 *)v10[1];
            v9 = (__int64 *)v13;
            if ( v24 == (__int64 *)v10 )
              goto LABEL_30;
          }
        }
      }
    }
  }
}
