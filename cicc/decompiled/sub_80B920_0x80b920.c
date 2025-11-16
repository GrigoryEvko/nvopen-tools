// Function: sub_80B920
// Address: 0x80b920
//
void __fastcall sub_80B920(__int64 *a1, _QWORD *a2)
{
  __int64 *v2; // rbx
  __int64 v3; // rax
  const char *v4; // r12
  __int64 *v5; // r15
  __int64 **v6; // r14
  _QWORD *j; // rbx
  __int64 v8; // rax
  const char *v9; // r15
  size_t v10; // rax
  __int64 v11; // rdx
  size_t v12; // rax
  _QWORD *v13; // r12
  __int64 v14; // rax
  int v15; // eax
  __int64 *v16; // [rsp+8h] [rbp-88h]
  __int64 v17; // [rsp+10h] [rbp-80h]
  _QWORD *i; // [rsp+18h] [rbp-78h]
  _WORD v19[56]; // [rsp+20h] [rbp-70h] BYREF

  v16 = a1;
  if ( a1 )
  {
    v17 = 0;
    do
    {
      if ( *((_BYTE *)v16 + 8) == 80 )
      {
        for ( i = (_QWORD *)v16[4]; i; i = (_QWORD *)*i )
        {
          v2 = (__int64 *)qword_4F18BA8;
          if ( qword_4F18BA8 )
            qword_4F18BA8 = *(_QWORD *)qword_4F18BA8;
          else
            v2 = (__int64 *)sub_822B10(16);
          v3 = i[5];
          v2[1] = v3;
          if ( v17 )
          {
            v4 = *(const char **)(v3 + 184);
            v5 = (__int64 *)v17;
            v6 = 0;
            while ( strcmp(v4, *(const char **)(v5[1] + 184)) > 0 )
            {
              v6 = (__int64 **)v5;
              if ( !*v5 )
              {
                *v5 = (__int64)v2;
                *v2 = 0;
                goto LABEL_8;
              }
              v5 = (__int64 *)*v5;
            }
            if ( (__int64 *)v17 == v5 )
            {
              *v2 = v17;
              v17 = (__int64)v2;
            }
            else
            {
              *v6 = v2;
              *v2 = (__int64)v5;
            }
          }
          else
          {
            *v2 = 0;
            v17 = (__int64)v2;
          }
LABEL_8:
          ;
        }
      }
      v16 = (__int64 *)*v16;
    }
    while ( v16 );
    if ( v17 )
    {
      for ( j = (_QWORD *)v17; ; j = (_QWORD *)*j )
      {
        v13 = (_QWORD *)qword_4F18BE0;
        ++*a2;
        v14 = v13[2];
        if ( (unsigned __int64)(v14 + 1) > v13[1] )
        {
          sub_823810(v13);
          v13 = (_QWORD *)qword_4F18BE0;
          v14 = *(_QWORD *)(qword_4F18BE0 + 16);
        }
        *(_BYTE *)(v13[4] + v14) = 66;
        ++v13[2];
        v8 = j[1];
        v9 = *(const char **)(v8 + 184);
        if ( v9 )
        {
          v10 = strlen(*(const char **)(v8 + 184));
          if ( v10 > 9 )
          {
            v15 = sub_622470(v10, v19);
            v13 = (_QWORD *)qword_4F18BE0;
            v11 = v15;
          }
          else
          {
            v11 = 1;
            v19[0] = (unsigned __int8)(v10 + 48);
          }
          *a2 += v11;
          sub_8238B0(v13, v19, v11);
          v12 = strlen(v9);
          *a2 += v12;
          sub_8238B0(qword_4F18BE0, v9, v12);
        }
        else
        {
          ++*a2;
          v19[0] = 48;
          sub_8238B0(v13, v19, 1);
        }
        if ( !*j )
          break;
      }
      *j = qword_4F18BA8;
      qword_4F18BA8 = v17;
    }
  }
}
