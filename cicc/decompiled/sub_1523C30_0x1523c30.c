// Function: sub_1523C30
// Address: 0x1523c30
//
size_t __fastcall sub_1523C30(__int64 a1, char *a2, char *a3, char *a4)
{
  signed __int64 v5; // r8
  unsigned __int64 v6; // r11
  char *v9; // r12
  __int64 v11; // rsi
  size_t v12; // rdi
  unsigned __int64 v13; // rcx
  size_t result; // rax
  char *v15; // r10
  char *v16; // r9
  char *v17; // r15
  size_t v18; // rdx
  __int64 v19; // rcx
  unsigned int v20; // esi
  size_t v21; // r11
  char *v22; // r15
  void *v23; // rdi
  size_t v24; // [rsp+0h] [rbp-60h]
  signed __int64 v25; // [rsp+8h] [rbp-58h]
  signed __int64 v26; // [rsp+8h] [rbp-58h]
  __int64 v27; // [rsp+10h] [rbp-50h]
  size_t v28; // [rsp+10h] [rbp-50h]
  size_t v29; // [rsp+10h] [rbp-50h]
  signed __int64 v30; // [rsp+18h] [rbp-48h]
  int v31; // [rsp+18h] [rbp-48h]
  char *v32; // [rsp+18h] [rbp-48h]
  char *v33; // [rsp+18h] [rbp-48h]
  char *v34; // [rsp+20h] [rbp-40h]
  signed __int64 v35; // [rsp+20h] [rbp-40h]
  char *v36; // [rsp+20h] [rbp-40h]
  signed __int64 v37; // [rsp+20h] [rbp-40h]
  char *v38; // [rsp+20h] [rbp-40h]
  size_t v39; // [rsp+28h] [rbp-38h]
  int v40; // [rsp+28h] [rbp-38h]
  char *v41; // [rsp+28h] [rbp-38h]
  __int64 v42; // [rsp+28h] [rbp-38h]
  signed __int64 v43; // [rsp+28h] [rbp-38h]
  int v44; // [rsp+28h] [rbp-38h]

  v5 = a4 - a3;
  v6 = (a4 - a3) >> 3;
  v9 = a2;
  v11 = *(unsigned int *)(a1 + 8);
  v12 = *(_QWORD *)a1;
  v13 = *(unsigned int *)(a1 + 12);
  result = 8 * v11;
  v15 = &a2[-v12];
  v16 = (char *)(v12 + 8 * v11);
  if ( v9 == v16 )
  {
    if ( v13 - v11 < v6 )
    {
      v37 = v5;
      v44 = v6;
      sub_16CD150(a1, a1 + 16, v11 + v6, 8);
      v11 = *(unsigned int *)(a1 + 8);
      result = *(_QWORD *)a1;
      v5 = v37;
      LODWORD(v6) = v44;
      v16 = (char *)(*(_QWORD *)a1 + 8 * v11);
    }
    if ( a3 != a4 )
    {
      v40 = v6;
      result = (size_t)memmove(v16, a3, v5);
      LODWORD(v11) = *(_DWORD *)(a1 + 8);
      LODWORD(v6) = v40;
    }
    *(_DWORD *)(a1 + 8) = v6 + v11;
  }
  else
  {
    v17 = a3;
    if ( v11 + v6 > v13 )
    {
      v31 = v6;
      v35 = v5;
      v41 = v15;
      sub_16CD150(a1, a1 + 16, v11 + v6, 8);
      v11 = *(unsigned int *)(a1 + 8);
      v12 = *(_QWORD *)a1;
      v15 = v41;
      LODWORD(v6) = v31;
      result = 8 * v11;
      v5 = v35;
      v9 = &v41[*(_QWORD *)a1];
      v16 = (char *)(*(_QWORD *)a1 + 8 * v11);
    }
    v18 = result - (_QWORD)v15;
    v19 = (__int64)(result - (_QWORD)v15) >> 3;
    if ( result - (unsigned __int64)v15 >= v5 )
    {
      v21 = result - v5;
      result = v5;
      v22 = (char *)(v12 + v21);
      v23 = v16;
      v42 = v5 >> 3;
      if ( v5 >> 3 > (unsigned __int64)*(unsigned int *)(a1 + 12) - v11 )
      {
        v24 = v5;
        v26 = v5;
        v29 = v21;
        v33 = v16;
        v38 = v15;
        sub_16CD150(a1, a1 + 16, (v5 >> 3) + v11, 8);
        v11 = *(unsigned int *)(a1 + 8);
        result = v24;
        v5 = v26;
        v21 = v29;
        v16 = v33;
        v23 = (void *)(*(_QWORD *)a1 + 8 * v11);
        v15 = v38;
      }
      if ( v22 != v16 )
      {
        v25 = v5;
        v28 = v21;
        v32 = v16;
        v36 = v15;
        result = (size_t)memmove(v23, v22, result);
        LODWORD(v11) = *(_DWORD *)(a1 + 8);
        v5 = v25;
        v21 = v28;
        v16 = v32;
        v15 = v36;
      }
      *(_DWORD *)(a1 + 8) = v42 + v11;
      if ( v22 != v9 )
      {
        v43 = v5;
        result = (size_t)memmove(&v16[-(v21 - (_QWORD)v15)], v9, v21 - (_QWORD)v15);
        v5 = v43;
      }
      if ( a3 != a4 )
        return (size_t)memmove(v9, a3, v5);
    }
    else
    {
      v20 = v6 + v11;
      *(_DWORD *)(a1 + 8) = v20;
      if ( v16 != v9 )
      {
        v27 = (__int64)(result - (_QWORD)v15) >> 3;
        v30 = v5;
        v34 = v16;
        v39 = result - (_QWORD)v15;
        result = (size_t)memcpy((void *)(v12 + 8LL * v20 - v18), v9, v18);
        v19 = v27;
        v5 = v30;
        v16 = v34;
        v18 = v39;
      }
      if ( v19 )
      {
        for ( result = 0; result != v19; ++result )
          *(_QWORD *)&v9[8 * result] = *(_QWORD *)&a3[8 * result];
        v17 = &a3[v18];
        v5 = a4 - &a3[v18];
      }
      if ( v17 != a4 )
        return (size_t)memmove(v16, v17, v5);
    }
  }
  return result;
}
