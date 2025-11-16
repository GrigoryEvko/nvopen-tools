// Function: sub_2FAF8F0
// Address: 0x2faf8f0
//
unsigned int *__fastcall sub_2FAF8F0(_QWORD *a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int *result; // rax
  unsigned int *v7; // r14
  __int64 v8; // r13
  __int64 v9; // rcx
  __int64 v10; // rdx
  unsigned int v11; // r12d
  __int64 v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r10
  __int64 v18; // r13
  unsigned int **v19; // rdx
  unsigned int **v20; // rcx
  __int64 v21; // rbx
  unsigned int **v22; // rdx
  unsigned int **v23; // rcx
  bool v24; // cf
  unsigned __int64 v25; // r10
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rdx
  unsigned __int64 v28; // r11
  unsigned __int64 v29; // [rsp+8h] [rbp-58h]
  unsigned __int64 v30; // [rsp+8h] [rbp-58h]
  unsigned int *v31; // [rsp+10h] [rbp-50h]
  unsigned int *v32; // [rsp+10h] [rbp-50h]
  unsigned __int64 v33; // [rsp+18h] [rbp-48h]
  unsigned __int64 v34; // [rsp+20h] [rbp-40h]
  unsigned int *v35; // [rsp+28h] [rbp-38h]

  result = &a2[a3];
  v35 = result;
  if ( result != a2 )
  {
    v7 = a2;
    do
    {
      v8 = *v7;
      v9 = (unsigned int)(2 * v8);
      result = *(unsigned int **)(a1[1] + 8LL);
      v10 = (unsigned int)(2 * v8 + 1);
      v11 = result[v9];
      v12 = result[v10];
      if ( (_DWORD)v12 != v11 )
      {
        sub_2FAF160((__int64)a1, v11, v10, v9, a5, a6);
        sub_2FAF160((__int64)a1, v12, v13, v14, v15, v16);
        v17 = v11;
        result = *(unsigned int **)(a1[17] + 8 * v8);
        v18 = a1[3] + 112LL * v11;
        if ( __CFADD__(*(_QWORD *)(v18 + 104), result) )
          *(_QWORD *)(v18 + 104) = -1;
        else
          *(_QWORD *)(v18 + 104) += result;
        v19 = *(unsigned int ***)(v18 + 24);
        v20 = &v19[2 * *(unsigned int *)(v18 + 32)];
        if ( v19 == v20 )
        {
LABEL_25:
          v27 = *(unsigned int *)(v18 + 32) + 1LL;
          v28 = v12 | v34 & 0xFFFFFFFF00000000LL;
          v34 = v28;
          if ( v27 > *(unsigned int *)(v18 + 36) )
          {
            v30 = v28;
            v32 = result;
            sub_C8D5F0(v18 + 24, (const void *)(v18 + 40), v27, 0x10u, a5, a6);
            v17 = v11;
            v28 = v30;
            result = v32;
            v20 = (unsigned int **)(*(_QWORD *)(v18 + 24) + 16LL * *(unsigned int *)(v18 + 32));
          }
          *v20 = result;
          v20[1] = (unsigned int *)v28;
          ++*(_DWORD *)(v18 + 32);
        }
        else
        {
          while ( (_DWORD)v12 != *((_DWORD *)v19 + 2) )
          {
            v19 += 2;
            if ( v20 == v19 )
              goto LABEL_25;
          }
          if ( __CFADD__(*v19, result) )
            *v19 = (unsigned int *)-1LL;
          else
            *v19 = (unsigned int *)((char *)result + (_QWORD)*v19);
        }
        v21 = a1[3] + 112 * v12;
        if ( __CFADD__(*(_QWORD *)(v21 + 104), result) )
          *(_QWORD *)(v21 + 104) = -1;
        else
          *(_QWORD *)(v21 + 104) += result;
        v22 = *(unsigned int ***)(v21 + 24);
        v23 = &v22[2 * *(unsigned int *)(v21 + 32)];
        if ( v22 == v23 )
        {
LABEL_22:
          v25 = v33 & 0xFFFFFFFF00000000LL | v17;
          v26 = *(unsigned int *)(v21 + 32) + 1LL;
          v33 = v25;
          if ( v26 > *(unsigned int *)(v21 + 36) )
          {
            v29 = v25;
            v31 = result;
            sub_C8D5F0(v21 + 24, (const void *)(v21 + 40), v26, 0x10u, a5, a6);
            v25 = v29;
            result = v31;
            v23 = (unsigned int **)(*(_QWORD *)(v21 + 24) + 16LL * *(unsigned int *)(v21 + 32));
          }
          *v23 = result;
          v23[1] = (unsigned int *)v25;
          ++*(_DWORD *)(v21 + 32);
        }
        else
        {
          while ( v11 != *((_DWORD *)v22 + 2) )
          {
            v22 += 2;
            if ( v23 == v22 )
              goto LABEL_22;
          }
          v24 = __CFADD__(*v22, result);
          result = (unsigned int *)((char *)result + (_QWORD)*v22);
          if ( v24 )
            *v22 = (unsigned int *)-1LL;
          else
            *v22 = result;
        }
      }
      ++v7;
    }
    while ( v35 != v7 );
  }
  return result;
}
