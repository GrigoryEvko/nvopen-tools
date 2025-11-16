// Function: sub_D78B00
// Address: 0xd78b00
//
__int64 __fastcall sub_D78B00(_QWORD *a1, __int64 a2, size_t a3)
{
  __int64 result; // rax
  __int64 *v4; // r15
  const void *v5; // rsi
  __int64 v6; // r13
  __int64 v7; // rax
  void *v8; // rdi
  __int64 v9; // rbx
  __int64 v10; // r13
  size_t v11; // r8
  __int64 v12; // r12
  __int64 v13; // rax
  _QWORD *v14; // r14
  size_t v15; // [rsp-48h] [rbp-48h]
  __int64 *v16; // [rsp-40h] [rbp-40h]

  result = *((unsigned int *)a1 + 6);
  if ( (_DWORD)result )
  {
    v4 = (__int64 *)a1[1];
    v16 = &v4[5 * result];
    do
    {
      result = v4[3];
      v5 = (const void *)v4[2];
      v9 = *v4;
      v10 = v4[1];
      v11 = result - (_QWORD)v5;
      v12 = result - (_QWORD)v5;
      if ( result == (_QWORD)v5 )
      {
        v14 = 0;
      }
      else
      {
        if ( v11 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_28:
          sub_4261EA(a1, v5, a3);
        a1 = (_QWORD *)(v4[3] - (_QWORD)v5);
        v13 = sub_22077B0(v11);
        v5 = (const void *)v4[2];
        v14 = (_QWORD *)v13;
        result = v4[3];
        v11 = result - (_QWORD)v5;
      }
      if ( (const void *)result != v5 )
      {
        a1 = v14;
        v15 = v11;
        result = (__int64)memmove(v14, v5, v11);
        v11 = v15;
      }
      if ( !(v11 | v9) && v10 == -1 )
        goto LABEL_8;
      result = v4[3];
      v5 = (const void *)v4[2];
      a3 = result - (_QWORD)v5;
      v6 = result - (_QWORD)v5;
      if ( result == (_QWORD)v5 )
      {
        v8 = 0;
        if ( (const void *)result == v5 )
          goto LABEL_8;
      }
      else
      {
        if ( a3 > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_28;
        v7 = sub_22077B0(a3);
        v5 = (const void *)v4[2];
        v8 = (void *)v7;
        result = v4[3];
        a3 = result - (_QWORD)v5;
        if ( (const void *)result == v5 )
        {
          if ( !v8 )
            goto LABEL_8;
          goto LABEL_7;
        }
      }
      v8 = memmove(v8, v5, a3);
LABEL_7:
      result = j_j___libc_free_0(v8, v6);
LABEL_8:
      if ( v14 )
        result = j_j___libc_free_0(v14, v12);
      a1 = (_QWORD *)v4[2];
      if ( a1 )
        result = j_j___libc_free_0(a1, v4[4] - (_QWORD)a1);
      v4 += 5;
    }
    while ( v16 != v4 );
  }
  return result;
}
