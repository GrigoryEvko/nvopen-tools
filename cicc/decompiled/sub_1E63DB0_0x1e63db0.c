// Function: sub_1E63DB0
// Address: 0x1e63db0
//
_QWORD *__fastcall sub_1E63DB0(_QWORD *a1, _QWORD *a2, _QWORD *a3)
{
  __int64 v5; // r15
  _QWORD *result; // rax
  unsigned __int64 *v7; // r14
  unsigned __int64 *v8; // r12
  _QWORD *v9; // r8
  unsigned __int64 v10; // rsi
  _QWORD *v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rdx
  _QWORD *v14; // [rsp+0h] [rbp-40h]
  unsigned __int64 v15[7]; // [rsp+8h] [rbp-38h] BYREF

  v15[0] = (unsigned __int64)a2;
  v5 = a1[4];
  sub_1E63C90(a3, v15);
  sub_1E62C20(a1, (_QWORD *)v15[0]);
  result = (_QWORD *)v15[0];
  v7 = *(unsigned __int64 **)(v15[0] + 96);
  v8 = *(unsigned __int64 **)(v15[0] + 88);
  if ( v7 != v8 )
  {
    v9 = a3 + 1;
    do
    {
      v10 = *v8;
      if ( *v8 != v5 )
      {
        result = (_QWORD *)a3[2];
        if ( !result )
          goto LABEL_11;
        v11 = v9;
        do
        {
          while ( 1 )
          {
            v12 = result[2];
            v13 = result[3];
            if ( result[4] >= v10 )
              break;
            result = (_QWORD *)result[3];
            if ( !v13 )
              goto LABEL_9;
          }
          v11 = result;
          result = (_QWORD *)result[2];
        }
        while ( v12 );
LABEL_9:
        if ( v9 == v11 || v11[4] > v10 )
        {
LABEL_11:
          v14 = v9;
          result = (_QWORD *)sub_1E63DB0(a1, v10, a3);
          v9 = v14;
        }
      }
      ++v8;
    }
    while ( v7 != v8 );
  }
  return result;
}
