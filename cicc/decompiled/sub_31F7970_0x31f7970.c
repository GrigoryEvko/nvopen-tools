// Function: sub_31F7970
// Address: 0x31f7970
//
unsigned __int8 *__fastcall sub_31F7970(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 *v6; // r12
  unsigned __int8 *v8; // r14
  __int64 v9; // rdx
  const char *v10; // r8
  const char **v11; // rdx
  unsigned __int8 v12; // al
  __int64 v13; // rax
  const char *v14; // rcx
  const char *v15; // rdx
  const char *v16; // rdx
  const char *v18; // rax
  const char *v19; // [rsp+0h] [rbp-40h]
  const char *v20; // [rsp+8h] [rbp-38h]

  if ( !a2 )
    return 0;
  v6 = (unsigned __int8 *)a2;
  v8 = 0;
  do
  {
    while ( 1 )
    {
      v12 = *v6;
      if ( v8 || v12 != 18 )
      {
        if ( v12 == 14 )
        {
          v13 = *(unsigned int *)(a1 + 1288);
          if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 1292) )
          {
            a2 = a1 + 1296;
            sub_C8D5F0(a1 + 1280, (const void *)(a1 + 1296), v13 + 1, 8u, a5, a6);
            v13 = *(unsigned int *)(a1 + 1288);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 1280) + 8 * v13) = v6;
          ++*(_DWORD *)(a1 + 1288);
        }
      }
      else
      {
        v8 = v6;
      }
      v14 = sub_AF5A10(v6, a2);
      v18 = v15;
      if ( v15 )
        break;
      v14 = sub_31F3D90((__int64)v6);
      v18 = v16;
      if ( v16 )
        break;
      v6 = (unsigned __int8 *)sub_AF2660(v6);
      if ( !v6 )
        return v8;
    }
    v9 = *(unsigned int *)(a3 + 8);
    v10 = v14;
    if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
      a2 = a3 + 16;
      v19 = v14;
      v20 = v18;
      sub_C8D5F0(a3, (const void *)(a3 + 16), v9 + 1, 0x10u, (__int64)v14, v9 + 1);
      v9 = *(unsigned int *)(a3 + 8);
      v10 = v19;
      v18 = v20;
    }
    v11 = (const char **)(*(_QWORD *)a3 + 16 * v9);
    *v11 = v10;
    v11[1] = v18;
    ++*(_DWORD *)(a3 + 8);
    v6 = (unsigned __int8 *)sub_AF2660(v6);
  }
  while ( v6 );
  return v8;
}
