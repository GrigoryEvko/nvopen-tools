// Function: sub_9B8300
// Address: 0x9b8300
//
unsigned int *__fastcall sub_9B8300(unsigned int a1, unsigned int *a2, __int64 a3, __int64 a4)
{
  const void *v4; // r10
  signed __int64 v5; // r14
  unsigned int *result; // rax
  unsigned int *v8; // r14
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // r12
  int v12; // r9d
  int v13; // r13d
  __int64 v14; // rdx
  __int64 i; // [rsp+8h] [rbp-58h]
  unsigned int *v16; // [rsp+10h] [rbp-50h]
  __int64 v17; // [rsp+18h] [rbp-48h]
  unsigned int v18; // [rsp+24h] [rbp-3Ch]
  int v19; // [rsp+28h] [rbp-38h]

  v4 = a2;
  v5 = 4 * a3;
  if ( a1 == 1 )
  {
    *(_DWORD *)(a4 + 8) = 0;
    v14 = 0;
    result = 0;
    if ( v5 >> 2 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
    {
      sub_C8D5F0(a4, a4 + 16, v5 >> 2, 4);
      result = (unsigned int *)*(unsigned int *)(a4 + 8);
      v4 = a2;
      v14 = 4LL * (_QWORD)result;
    }
    if ( v5 )
    {
      memcpy((void *)(v14 + *(_QWORD *)a4), v4, v5);
      result = (unsigned int *)*(unsigned int *)(a4 + 8);
    }
    *(_DWORD *)(a4 + 8) = (_DWORD)result + (v5 >> 2);
  }
  else
  {
    result = &a2[(unsigned __int64)v5 / 4];
    v8 = a2;
    *(_DWORD *)(a4 + 8) = 0;
    v16 = result;
    for ( i = a4 + 16; v16 != v8; ++v8 )
    {
      result = (unsigned int *)*v8;
      if ( a1 )
      {
        v9 = *(unsigned int *)(a4 + 8);
        v10 = a1;
        v11 = 0;
        v12 = a1 * (_DWORD)result;
        do
        {
          v13 = v12 + v11;
          if ( (int)result < 0 )
            v13 = (int)result;
          if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
          {
            v17 = v10;
            v18 = (unsigned int)result;
            v19 = v12;
            sub_C8D5F0(a4, i, v9 + 1, 4);
            v9 = *(unsigned int *)(a4 + 8);
            v10 = v17;
            result = (unsigned int *)v18;
            v12 = v19;
          }
          ++v11;
          *(_DWORD *)(*(_QWORD *)a4 + 4 * v9) = v13;
          v9 = (unsigned int)(*(_DWORD *)(a4 + 8) + 1);
          *(_DWORD *)(a4 + 8) = v9;
        }
        while ( v11 != v10 );
      }
    }
  }
  return result;
}
