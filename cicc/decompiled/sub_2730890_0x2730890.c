// Function: sub_2730890
// Address: 0x2730890
//
unsigned __int8 **__fastcall sub_2730890(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 **v3; // r13
  unsigned __int8 **result; // rax
  unsigned __int8 *v6; // r14
  __int64 v7; // rbx
  unsigned __int8 *v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int64 v11; // rcx
  unsigned __int8 *v12; // r10
  unsigned __int64 v13; // rax
  unsigned __int8 *v14; // rdx
  unsigned __int8 *v15; // r11
  int v16; // edx
  unsigned __int8 *v17; // [rsp+0h] [rbp-50h]
  unsigned __int8 *v18; // [rsp+8h] [rbp-48h]
  const void *v19; // [rsp+10h] [rbp-40h]
  unsigned __int8 **v20; // [rsp+18h] [rbp-38h]

  v3 = *(unsigned __int8 ***)a2;
  v19 = (const void *)(a3 + 16);
  result = (unsigned __int8 **)(*(_QWORD *)a2 + 160LL * *(unsigned int *)(a2 + 8));
  v20 = result;
  if ( result != *(unsigned __int8 ***)a2 )
  {
    do
    {
      v6 = *v3;
      v7 = (__int64)&(*v3)[16 * *((unsigned int *)v3 + 2)];
      if ( (unsigned __int8 *)v7 != *v3 )
      {
        do
        {
          v8 = sub_27306B0(a1, *(unsigned __int8 **)v6, *((_DWORD *)v6 + 2));
          v11 = *(unsigned int *)(a3 + 12);
          v12 = v8;
          v13 = *(unsigned int *)(a3 + 8);
          v15 = v14;
          v16 = *(_DWORD *)(a3 + 8);
          if ( v13 >= v11 )
          {
            if ( v11 < v13 + 1 )
            {
              v17 = v12;
              v18 = v15;
              sub_C8D5F0(a3, v19, v13 + 1, 0x10u, v9, v10);
              v13 = *(unsigned int *)(a3 + 8);
              v12 = v17;
              v15 = v18;
            }
            result = (unsigned __int8 **)(*(_QWORD *)a3 + 16 * v13);
            *result = v12;
            result[1] = v15;
            ++*(_DWORD *)(a3 + 8);
          }
          else
          {
            result = (unsigned __int8 **)(*(_QWORD *)a3 + 16 * v13);
            if ( result )
            {
              *result = v12;
              result[1] = v15;
              v16 = *(_DWORD *)(a3 + 8);
            }
            *(_DWORD *)(a3 + 8) = v16 + 1;
          }
          v6 += 16;
        }
        while ( (unsigned __int8 *)v7 != v6 );
      }
      v3 += 20;
    }
    while ( v20 != v3 );
  }
  return result;
}
