// Function: sub_18341E0
// Address: 0x18341e0
//
_QWORD *__fastcall sub_18341E0(__int64 a1, unsigned __int64 *a2)
{
  _QWORD *result; // rax
  _QWORD *v5; // rdx
  _QWORD *v6; // r13
  _BOOL4 v7; // r8d
  _QWORD *v8; // rsi
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rax
  char *v13; // rdi
  char *v14; // rcx
  char *v15; // rax
  char *v16; // rdx
  _BOOL4 v17; // [rsp+Ch] [rbp-34h]

  result = sub_1834080(a1, a2);
  if ( v5 )
  {
    v6 = v5;
    v7 = 1;
    if ( !result && v5 != (_QWORD *)(a1 + 8) )
    {
      v12 = v5[4];
      if ( *a2 >= v12 )
      {
        v7 = 0;
        if ( *a2 == v12 )
        {
          v13 = (char *)v5[6];
          v14 = (char *)a2[2];
          v15 = (char *)a2[1];
          v16 = (char *)v5[5];
          if ( v14 - v15 > v13 - v16 )
            v14 = &v15[v13 - v16];
          if ( v15 == v14 )
          {
LABEL_16:
            v7 = v13 != v16;
          }
          else
          {
            while ( 1 )
            {
              if ( *(_QWORD *)v15 < *(_QWORD *)v16 )
              {
                v7 = 1;
                goto LABEL_3;
              }
              if ( *(_QWORD *)v15 > *(_QWORD *)v16 )
                break;
              v15 += 8;
              v16 += 8;
              if ( v14 == v15 )
                goto LABEL_16;
            }
            v7 = 0;
          }
        }
      }
    }
LABEL_3:
    v17 = v7;
    v8 = (_QWORD *)sub_22077B0(64);
    v8[4] = *a2;
    v9 = a2[1];
    a2[1] = 0;
    v8[5] = v9;
    v10 = a2[2];
    a2[2] = 0;
    v8[6] = v10;
    v11 = a2[3];
    a2[3] = 0;
    v8[7] = v11;
    sub_220F040(v17, v8, v6, a1 + 8);
    ++*(_QWORD *)(a1 + 40);
    return v8;
  }
  return result;
}
