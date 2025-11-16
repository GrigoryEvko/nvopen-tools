// Function: sub_130CA40
// Address: 0x130ca40
//
unsigned __int64 __fastcall sub_130CA40(void *a1, size_t a2, __int64 a3, _BYTE *a4)
{
  unsigned __int64 result; // rax
  __int64 v7; // r14
  size_t v8; // r15
  __int64 v9; // r13
  _BYTE *v10; // rax
  _BYTE *v11; // rdi
  size_t v12; // r8
  size_t v13; // [rsp+8h] [rbp-48h]
  unsigned __int64 v14; // [rsp+10h] [rbp-40h]
  unsigned __int64 v15; // [rsp+10h] [rbp-40h]
  __int64 v16; // [rsp+18h] [rbp-38h]

  result = (unsigned __int64)sub_130C9C0(a1, a2, a4);
  if ( result )
  {
    if ( a1 != (void *)result )
    {
      v7 = a3 - 1;
      if ( ((a3 - 1) & result) != 0 )
      {
        sub_130C960((void *)result, a2);
        v16 = a3 - qword_4F969C8;
        v8 = a2 + a3 - qword_4F969C8;
        if ( __CFADD__(a2, a3 - qword_4F969C8) )
          return 0;
        v9 = -a3;
        do
        {
          v10 = sub_130C9C0(0, v8, a4);
          v11 = v10;
          if ( !v10 )
            return 0;
          result = v9 & (unsigned __int64)&v10[v7];
          v12 = (size_t)&v11[v16 - result];
          if ( (_BYTE *)result == v11 )
          {
            if ( v12 )
              goto LABEL_12;
          }
          else
          {
            v13 = (size_t)&v11[v16 - result];
            v14 = result;
            sub_130C960(v11, result - (_QWORD)v11);
            v12 = v13;
            result = v14;
            if ( v13 )
            {
LABEL_12:
              v15 = result;
              sub_130C960((void *)(a2 + result), v12);
              result = v15;
            }
          }
        }
        while ( !result );
      }
    }
  }
  return result;
}
