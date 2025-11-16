// Function: sub_2CC92F0
// Address: 0x2cc92f0
//
unsigned __int64 __fastcall sub_2CC92F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 a6)
{
  unsigned __int64 result; // rax
  unsigned __int64 v8; // r14
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // r12
  unsigned __int64 v13; // r13
  char v14; // al
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r8
  __int64 v19; // rdi
  __int64 v20; // r8
  __int64 v21; // r9
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rcx
  int v24; // edx
  __int64 *v25; // rax
  __int64 *v26; // rax
  unsigned __int64 v29; // [rsp+28h] [rbp-58h]
  __int64 v30; // [rsp+28h] [rbp-58h]
  __int64 v31; // [rsp+30h] [rbp-50h]
  __int64 v32; // [rsp+30h] [rbp-50h]
  unsigned __int64 v34; // [rsp+40h] [rbp-40h] BYREF
  __int64 v35; // [rsp+48h] [rbp-38h]

  result = sub_AE4AC0(a2, a1);
  if ( *(_DWORD *)(a1 + 12) )
  {
    v8 = result + 24;
    v9 = 0;
    v10 = a6;
    v11 = v8;
    v13 = v10;
    do
    {
      v14 = *(_BYTE *)(v11 + 8);
      v34 = *(_QWORD *)v11;
      LOBYTE(v35) = v14;
      v29 = v13;
      v13 = sub_CA1930(&v34) + a4;
      v15 = sub_9208B0(a2, *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8 * v9));
      v35 = v16;
      v31 = a5;
      v34 = (unsigned __int64)(v15 + 7) >> 3;
      v17 = sub_CA1930(&v34);
      v18 = a5;
      a5 = v17;
      v19 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8 * v9);
      if ( *(_BYTE *)(v19 + 8) == 15 )
      {
        sub_2CC92F0(v19, a2, a3, v13, v18, v29);
      }
      else
      {
        v20 = v29 + v31;
        if ( v29 + v31 < v13 )
        {
          v21 = v13 - v20;
          v22 = *(unsigned int *)(a3 + 8);
          v23 = *(unsigned int *)(a3 + 12);
          v24 = *(_DWORD *)(a3 + 8);
          if ( v22 >= v23 )
          {
            if ( v23 < v22 + 1 )
            {
              v30 = v13 - v20;
              v32 = v20;
              sub_C8D5F0(a3, (const void *)(a3 + 16), v22 + 1, 0x10u, v20, v21);
              v21 = v30;
              v20 = v32;
              v22 = *(unsigned int *)(a3 + 8);
            }
            v26 = (__int64 *)(*(_QWORD *)a3 + 16 * v22);
            *v26 = v20;
            v26[1] = v21;
            ++*(_DWORD *)(a3 + 8);
          }
          else
          {
            v25 = (__int64 *)(*(_QWORD *)a3 + 16 * v22);
            if ( v25 )
            {
              *v25 = v20;
              v25[1] = v21;
              v24 = *(_DWORD *)(a3 + 8);
            }
            *(_DWORD *)(a3 + 8) = v24 + 1;
          }
        }
      }
      result = *(unsigned int *)(a1 + 12);
      ++v9;
      v11 += 16LL;
    }
    while ( result > v9 );
  }
  return result;
}
