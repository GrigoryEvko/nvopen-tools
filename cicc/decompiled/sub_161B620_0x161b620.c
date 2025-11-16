// Function: sub_161B620
// Address: 0x161b620
//
void __fastcall sub_161B620(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  char *v6; // r15
  char *v7; // r9
  __int64 v8; // r12
  __int64 v9; // r13
  char *v10; // r9
  char *v11; // r10
  char *v12; // r11
  __int64 v13; // r14
  char *v14; // rax
  char *v15; // r10
  __int64 v16; // rdx
  char *v18; // [rsp+8h] [rbp-48h]
  char *v19; // [rsp+10h] [rbp-40h]
  char *v20; // [rsp+18h] [rbp-38h]

  if ( a5 )
  {
    v5 = a4;
    if ( a4 )
    {
      v6 = a1;
      v7 = a2;
      v8 = a5;
      if ( a4 + a5 == 2 )
      {
        v15 = a2;
        v14 = a1;
LABEL_12:
        v16 = *(_QWORD *)v14;
        if ( *(_QWORD *)v14 > *(_QWORD *)v15 )
        {
          *(_QWORD *)v14 = *(_QWORD *)v15;
          *(_QWORD *)v15 = v16;
        }
      }
      else
      {
        if ( a5 >= a4 )
          goto LABEL_10;
LABEL_5:
        v9 = v5 / 2;
        v11 = (char *)sub_161B3C0(v7, a3, &v6[8 * (v5 / 2)]);
        v13 = (v11 - v10) >> 3;
        while ( 1 )
        {
          v18 = v11;
          v20 = v12;
          v8 -= v13;
          v19 = sub_161B460(v12, v10, v11);
          sub_161B620(v6, v20, v19, v9, v13);
          v5 -= v9;
          if ( !v5 )
            break;
          v14 = v19;
          v15 = v18;
          if ( !v8 )
            break;
          if ( v8 + v5 == 2 )
            goto LABEL_12;
          v7 = v18;
          v6 = v19;
          if ( v8 < v5 )
            goto LABEL_5;
LABEL_10:
          v13 = v8 / 2;
          v12 = (char *)sub_161B410(v6, (__int64)v7, &v7[8 * (v8 / 2)]);
          v9 = (v12 - v6) >> 3;
        }
      }
    }
  }
}
