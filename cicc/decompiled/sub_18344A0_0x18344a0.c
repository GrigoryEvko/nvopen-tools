// Function: sub_18344A0
// Address: 0x18344a0
//
_QWORD *__fastcall sub_18344A0(__int64 a1, _QWORD *a2)
{
  _QWORD *result; // rax
  __int64 v4; // rdx
  __int64 v5; // r14
  _BOOL4 v6; // r8d
  _QWORD *v7; // rbx
  _QWORD *v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  char *v11; // rdi
  _QWORD *v12; // rcx
  _QWORD *v13; // rax
  char *v14; // rdx
  _BOOL4 v15; // [rsp+Ch] [rbp-34h]

  result = sub_1834350(a1, (__int64)a2);
  if ( v4 )
  {
    v5 = v4;
    v6 = 1;
    if ( !result && v4 != a1 + 8 )
    {
      v11 = *(char **)(v4 + 40);
      v12 = (_QWORD *)a2[1];
      v13 = (_QWORD *)*a2;
      v14 = *(char **)(v4 + 32);
      if ( (__int64)v12 - *a2 > v11 - v14 )
        v12 = (_QWORD *)(*a2 + v11 - v14);
      if ( v13 == v12 )
      {
LABEL_14:
        v6 = v11 != v14;
      }
      else
      {
        while ( 1 )
        {
          if ( *v13 < *(_QWORD *)v14 )
          {
            v6 = 1;
            goto LABEL_3;
          }
          if ( *v13 > *(_QWORD *)v14 )
            break;
          ++v13;
          v14 += 8;
          if ( v12 == v13 )
            goto LABEL_14;
        }
        v6 = 0;
      }
    }
LABEL_3:
    v15 = v6;
    v7 = (_QWORD *)sub_22077B0(56);
    v8 = (_QWORD *)*a2;
    *a2 = 0;
    v7[4] = v8;
    v9 = a2[1];
    a2[1] = 0;
    v7[5] = v9;
    v10 = a2[2];
    a2[2] = 0;
    v7[6] = v10;
    sub_220F040(v15, v7, v5, a1 + 8);
    ++*(_QWORD *)(a1 + 40);
    return v7;
  }
  return result;
}
