// Function: sub_1F12980
// Address: 0x1f12980
//
__int64 *__fastcall sub_1F12980(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 *result; // rax
  unsigned int *v8; // rbx
  __int64 v9; // r12
  __int64 v10; // rcx
  __int64 v11; // rdx
  unsigned int v12; // r14d
  unsigned int v13; // r13d
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // r8d
  int v17; // r9d
  __int64 v18; // r12
  __int64 v19; // r9
  __int64 v20; // rdi
  __int64 v21; // rcx
  __int64 *v22; // rax
  __int64 v23; // r8
  __int64 v24; // r13
  __int64 v25; // rdi
  __int64 v26; // rdx
  __int64 v27; // r11
  unsigned int *v28; // [rsp+20h] [rbp-40h]
  __int64 v29; // [rsp+28h] [rbp-38h]

  result = (__int64 *)((char *)a2 + 4 * a3);
  v28 = (unsigned int *)result;
  if ( result != a2 )
  {
    v8 = (unsigned int *)a2;
    do
    {
      v9 = *v8;
      v10 = (unsigned int)(2 * v9);
      result = *(__int64 **)(a1[30] + 240LL);
      v11 = (unsigned int)(2 * v9 + 1);
      v12 = *((_DWORD *)result + v10);
      v13 = *((_DWORD *)result + v11);
      if ( v13 != v12 )
      {
        sub_1F12210((__int64)a1, v12, v11, v10, a5, a6);
        sub_1F12210((__int64)a1, v13, v14, v15, v16, v17);
        v18 = *(_QWORD *)(a1[47] + 8 * v9);
        v29 = a1[33] + 112LL * v12;
        sub_16AF570((_QWORD *)(v29 + 104), v18);
        v19 = v29;
        v20 = *(_QWORD *)(v29 + 24);
        v21 = *(unsigned int *)(v29 + 32);
        v22 = (__int64 *)(v20 + 16 * v21);
        if ( (__int64 *)v20 == v22 )
        {
LABEL_19:
          v23 = v13;
          v27 = v13;
          if ( (unsigned int)v21 >= *(_DWORD *)(v29 + 36) )
          {
            sub_16CD150(v29 + 24, (const void *)(v29 + 40), 0, 16, v13, v29);
            v19 = v29;
            v23 = v13;
            v27 = v13;
            v22 = (__int64 *)(*(_QWORD *)(v29 + 24) + 16LL * *(unsigned int *)(v29 + 32));
          }
          *v22 = v18;
          v22[1] = v27;
          ++*(_DWORD *)(v19 + 32);
        }
        else
        {
          while ( v13 != *(_DWORD *)(v20 + 8) )
          {
            v20 += 16;
            if ( v22 == (__int64 *)v20 )
              goto LABEL_19;
          }
          sub_16AF570((_QWORD *)v20, v18);
          v23 = v13;
        }
        v24 = a1[33] + 112 * v23;
        sub_16AF570((_QWORD *)(v24 + 104), v18);
        v25 = *(_QWORD *)(v24 + 24);
        v26 = *(unsigned int *)(v24 + 32);
        result = (__int64 *)(v25 + 16 * v26);
        if ( (__int64 *)v25 == result )
        {
LABEL_16:
          if ( (unsigned int)v26 >= *(_DWORD *)(v24 + 36) )
          {
            sub_16CD150(v24 + 24, (const void *)(v24 + 40), 0, 16, a5, a6);
            result = (__int64 *)(*(_QWORD *)(v24 + 24) + 16LL * *(unsigned int *)(v24 + 32));
          }
          *result = v18;
          result[1] = v12;
          ++*(_DWORD *)(v24 + 32);
        }
        else
        {
          while ( v12 != *(_DWORD *)(v25 + 8) )
          {
            v25 += 16;
            if ( result == (__int64 *)v25 )
              goto LABEL_16;
          }
          result = sub_16AF570((_QWORD *)v25, v18);
        }
      }
      ++v8;
    }
    while ( v8 != v28 );
  }
  return result;
}
