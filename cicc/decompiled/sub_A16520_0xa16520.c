// Function: sub_A16520
// Address: 0xa16520
//
size_t __fastcall sub_A16520(__int64 a1, char *a2, char *a3, char *a4)
{
  signed __int64 v5; // r8
  __int64 v6; // r11
  char *v7; // r15
  char *v10; // r12
  __int64 v12; // rsi
  size_t v13; // rdi
  unsigned __int64 v14; // rcx
  size_t result; // rax
  char *v16; // r10
  unsigned __int64 v17; // rdx
  char *v18; // r9
  size_t v19; // rdx
  __int64 v20; // rcx
  unsigned int v21; // esi
  size_t v22; // r11
  char *v23; // r15
  unsigned __int64 v24; // rdx
  void *v25; // rdi
  size_t v26; // [rsp+0h] [rbp-60h]
  signed __int64 v27; // [rsp+8h] [rbp-58h]
  signed __int64 v28; // [rsp+8h] [rbp-58h]
  __int64 v29; // [rsp+10h] [rbp-50h]
  size_t v30; // [rsp+10h] [rbp-50h]
  size_t v31; // [rsp+10h] [rbp-50h]
  signed __int64 v32; // [rsp+18h] [rbp-48h]
  int v33; // [rsp+18h] [rbp-48h]
  char *v34; // [rsp+18h] [rbp-48h]
  char *v35; // [rsp+18h] [rbp-48h]
  char *v36; // [rsp+20h] [rbp-40h]
  signed __int64 v37; // [rsp+20h] [rbp-40h]
  char *v38; // [rsp+20h] [rbp-40h]
  int v39; // [rsp+20h] [rbp-40h]
  char *v40; // [rsp+20h] [rbp-40h]
  size_t v41; // [rsp+28h] [rbp-38h]
  int v42; // [rsp+28h] [rbp-38h]
  char *v43; // [rsp+28h] [rbp-38h]
  __int64 v44; // [rsp+28h] [rbp-38h]
  signed __int64 v45; // [rsp+28h] [rbp-38h]
  signed __int64 v46; // [rsp+28h] [rbp-38h]

  v5 = a4 - a3;
  v6 = (a4 - a3) >> 3;
  v7 = a3;
  v10 = a2;
  v12 = *(unsigned int *)(a1 + 8);
  v13 = *(_QWORD *)a1;
  v14 = *(unsigned int *)(a1 + 12);
  result = 8 * v12;
  v16 = &a2[-v13];
  v17 = v12 + v6;
  v18 = (char *)(v13 + 8 * v12);
  if ( v10 == v18 )
  {
    if ( v17 > v14 )
    {
      v39 = v6;
      v46 = v5;
      sub_C8D5F0(a1, a1 + 16, v17, 8);
      v12 = *(unsigned int *)(a1 + 8);
      result = *(_QWORD *)a1;
      LODWORD(v6) = v39;
      v5 = v46;
      v18 = (char *)(*(_QWORD *)a1 + 8 * v12);
    }
    if ( a3 != a4 )
    {
      v42 = v6;
      result = (size_t)memmove(v18, a3, v5);
      LODWORD(v12) = *(_DWORD *)(a1 + 8);
      LODWORD(v6) = v42;
    }
    *(_DWORD *)(a1 + 8) = v6 + v12;
  }
  else
  {
    if ( v17 > v14 )
    {
      v33 = v6;
      v37 = v5;
      v43 = v16;
      sub_C8D5F0(a1, a1 + 16, v17, 8);
      v12 = *(unsigned int *)(a1 + 8);
      v13 = *(_QWORD *)a1;
      v16 = v43;
      LODWORD(v6) = v33;
      result = 8 * v12;
      v5 = v37;
      v10 = &v43[*(_QWORD *)a1];
      v18 = (char *)(*(_QWORD *)a1 + 8 * v12);
    }
    v19 = result - (_QWORD)v16;
    v20 = (__int64)(result - (_QWORD)v16) >> 3;
    if ( result - (unsigned __int64)v16 >= v5 )
    {
      v22 = result - v5;
      result = v5;
      v23 = (char *)(v13 + v22);
      v24 = (v5 >> 3) + v12;
      v44 = v5 >> 3;
      v25 = v18;
      if ( v24 > *(unsigned int *)(a1 + 12) )
      {
        v26 = v5;
        v28 = v5;
        v31 = v22;
        v35 = v18;
        v40 = v16;
        sub_C8D5F0(a1, a1 + 16, v24, 8);
        v12 = *(unsigned int *)(a1 + 8);
        result = v26;
        v5 = v28;
        v22 = v31;
        v18 = v35;
        v25 = (void *)(*(_QWORD *)a1 + 8 * v12);
        v16 = v40;
      }
      if ( v23 != v18 )
      {
        v27 = v5;
        v30 = v22;
        v34 = v18;
        v38 = v16;
        result = (size_t)memmove(v25, v23, result);
        LODWORD(v12) = *(_DWORD *)(a1 + 8);
        v5 = v27;
        v22 = v30;
        v18 = v34;
        v16 = v38;
      }
      *(_DWORD *)(a1 + 8) = v44 + v12;
      if ( v23 != v10 )
      {
        v45 = v5;
        result = (size_t)memmove(&v18[-(v22 - (_QWORD)v16)], v10, v22 - (_QWORD)v16);
        v5 = v45;
      }
      if ( a3 != a4 )
        return (size_t)memmove(v10, a3, v5);
    }
    else
    {
      v21 = v6 + v12;
      *(_DWORD *)(a1 + 8) = v21;
      if ( v18 != v10 )
      {
        v29 = (__int64)(result - (_QWORD)v16) >> 3;
        v32 = v5;
        v36 = v18;
        v41 = result - (_QWORD)v16;
        result = (size_t)memcpy((void *)(v13 + 8LL * v21 - v19), v10, v19);
        v20 = v29;
        v5 = v32;
        v18 = v36;
        v19 = v41;
      }
      if ( v20 )
      {
        for ( result = 0; result != v20; ++result )
          *(_QWORD *)&v10[8 * result] = *(_QWORD *)&a3[8 * result];
        v7 = &a3[v19];
        v5 = a4 - &a3[v19];
      }
      if ( v7 != a4 )
        return (size_t)memmove(v18, v7, v5);
    }
  }
  return result;
}
