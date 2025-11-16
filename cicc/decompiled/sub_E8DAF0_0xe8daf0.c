// Function: sub_E8DAF0
// Address: 0xe8daf0
//
__int64 *__fastcall sub_E8DAF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *result; // rax
  __int64 v8; // rdi
  __int64 v9; // r8
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 *v12; // rdi
  __int64 *v13; // r13
  __int64 *v14; // rbx
  __int64 v15; // rsi
  __int64 v16; // r14
  __int64 v17; // r15
  __int64 (__fastcall *v18)(__int64, __int64, __int64); // rax
  __int64 *v19; // [rsp+8h] [rbp-38h]

  result = (__int64 *)*(unsigned int *)(a1 + 432);
  v8 = *(_QWORD *)(a1 + 416);
  if ( (_DWORD)result )
  {
    v9 = (unsigned int)((_DWORD)result - 1);
    v10 = (unsigned int)v9 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v19 = (__int64 *)(v8 + 40 * v10);
    v11 = *v19;
    if ( a2 == *v19 )
    {
LABEL_3:
      result = (__int64 *)(v8 + 40LL * (_QWORD)result);
      if ( v19 != result )
      {
        v12 = (__int64 *)v19[1];
        v13 = &v12[2 * *((unsigned int *)v19 + 4)];
        if ( v13 != v12 )
        {
          v14 = (__int64 *)v19[1];
          do
          {
            while ( 1 )
            {
              v16 = v14[1];
              v17 = *v14;
              v18 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a1 + 272LL);
              if ( v18 != sub_E8DCD0 )
                break;
              v15 = *v14;
              v14 += 2;
              sub_E5CB20(*(_QWORD *)(a1 + 296), v15, v10, (__int64)sub_E8DCD0, v9, a6);
              sub_E9A490(a1, v17, v16);
              a2 = v17;
              sub_E8DAF0(a1, v17);
              if ( v13 == v14 )
                goto LABEL_9;
            }
            v14 += 2;
            a2 = v17;
            v18(a1, v17, v16);
          }
          while ( v13 != v14 );
LABEL_9:
          v12 = (__int64 *)v19[1];
        }
        if ( v12 != v19 + 3 )
          _libc_free(v12, a2);
        result = v19;
        *v19 = -8192;
        --*(_DWORD *)(a1 + 424);
        ++*(_DWORD *)(a1 + 428);
      }
    }
    else
    {
      a6 = 1;
      while ( v11 != -4096 )
      {
        v10 = (unsigned int)v9 & ((_DWORD)a6 + (_DWORD)v10);
        v11 = *(_QWORD *)(v8 + 40LL * (unsigned int)v10);
        v19 = (__int64 *)(v8 + 40LL * (unsigned int)v10);
        if ( a2 == v11 )
          goto LABEL_3;
        a6 = (unsigned int)(a6 + 1);
      }
    }
  }
  return result;
}
