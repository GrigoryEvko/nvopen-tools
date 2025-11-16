// Function: sub_393D800
// Address: 0x393d800
//
size_t __fastcall sub_393D800(__int64 a1, char *a2, char *a3, char *a4)
{
  char *v5; // r13
  __int64 v6; // r9
  char *v8; // r12
  unsigned __int64 v10; // r11
  __int64 v11; // rcx
  size_t v12; // rsi
  unsigned __int64 v13; // rdx
  size_t result; // rax
  char *v15; // r10
  char *v16; // r8
  size_t v17; // rdx
  __int64 v18; // r15
  unsigned int v19; // ecx
  void *v20; // rdi
  size_t v21; // r11
  char *v22; // r15
  __int64 v23; // [rsp+0h] [rbp-60h]
  __int64 v24; // [rsp+8h] [rbp-58h]
  size_t v25; // [rsp+8h] [rbp-58h]
  size_t v26; // [rsp+10h] [rbp-50h]
  size_t v27; // [rsp+10h] [rbp-50h]
  int v28; // [rsp+18h] [rbp-48h]
  char *v29; // [rsp+18h] [rbp-48h]
  char *v30; // [rsp+18h] [rbp-48h]
  char *v31; // [rsp+20h] [rbp-40h]
  __int64 v32; // [rsp+20h] [rbp-40h]
  char *v33; // [rsp+20h] [rbp-40h]
  __int64 v34; // [rsp+20h] [rbp-40h]
  char *v35; // [rsp+20h] [rbp-40h]
  size_t v36; // [rsp+28h] [rbp-38h]
  int v37; // [rsp+28h] [rbp-38h]
  char *v38; // [rsp+28h] [rbp-38h]
  __int64 v39; // [rsp+28h] [rbp-38h]
  __int64 v40; // [rsp+28h] [rbp-38h]
  int v41; // [rsp+28h] [rbp-38h]

  v5 = a2;
  v6 = a4 - a3;
  v8 = a3;
  v10 = (a4 - a3) >> 3;
  v11 = *(unsigned int *)(a1 + 8);
  v12 = *(_QWORD *)a1;
  v13 = *(unsigned int *)(a1 + 12);
  result = 8 * v11;
  v15 = &a2[-*(_QWORD *)a1];
  v16 = (char *)(*(_QWORD *)a1 + 8 * v11);
  if ( v5 == v16 )
  {
    if ( v13 - v11 < v10 )
    {
      v34 = v6;
      v41 = v10;
      sub_16CD150(a1, (const void *)(a1 + 16), v11 + v10, 8, (int)v16, v6);
      v11 = *(unsigned int *)(a1 + 8);
      result = *(_QWORD *)a1;
      v6 = v34;
      LODWORD(v10) = v41;
      v16 = (char *)(*(_QWORD *)a1 + 8 * v11);
    }
    if ( v8 != a4 )
    {
      v37 = v10;
      result = (size_t)memcpy(v16, v8, v6);
      LODWORD(v11) = *(_DWORD *)(a1 + 8);
      LODWORD(v10) = v37;
    }
    *(_DWORD *)(a1 + 8) = v10 + v11;
  }
  else
  {
    if ( v11 + v10 > v13 )
    {
      v28 = v10;
      v32 = v6;
      v38 = v15;
      sub_16CD150(a1, (const void *)(a1 + 16), v11 + v10, 8, (int)v16, v6);
      v11 = *(unsigned int *)(a1 + 8);
      v12 = *(_QWORD *)a1;
      v15 = v38;
      LODWORD(v10) = v28;
      result = 8 * v11;
      v6 = v32;
      v5 = &v38[*(_QWORD *)a1];
      v16 = (char *)(*(_QWORD *)a1 + 8 * v11);
    }
    v17 = result - (_QWORD)v15;
    v18 = (__int64)(result - (_QWORD)v15) >> 3;
    if ( result - (unsigned __int64)v15 >= v6 )
    {
      v20 = v16;
      v21 = result - v6;
      result = v6;
      v22 = (char *)(v12 + v21);
      v39 = v6 >> 3;
      if ( v6 >> 3 > (unsigned __int64)*(unsigned int *)(a1 + 12) - v11 )
      {
        v23 = v6;
        v25 = v6;
        v27 = v21;
        v30 = v16;
        v35 = v15;
        sub_16CD150(a1, (const void *)(a1 + 16), v11 + (v6 >> 3), 8, (int)v16, v6);
        v11 = *(unsigned int *)(a1 + 8);
        v6 = v23;
        result = v25;
        v21 = v27;
        v16 = v30;
        v20 = (void *)(*(_QWORD *)a1 + 8 * v11);
        v15 = v35;
      }
      if ( v22 != v16 )
      {
        v24 = v6;
        v26 = v21;
        v29 = v16;
        v33 = v15;
        result = (size_t)memmove(v20, v22, result);
        LODWORD(v11) = *(_DWORD *)(a1 + 8);
        v6 = v24;
        v21 = v26;
        v16 = v29;
        v15 = v33;
      }
      *(_DWORD *)(a1 + 8) = v39 + v11;
      if ( v22 != v5 )
      {
        v40 = v6;
        result = (size_t)memmove(&v16[-(v21 - (_QWORD)v15)], v5, v21 - (_QWORD)v15);
        v6 = v40;
      }
      if ( v8 != a4 )
        return (size_t)memmove(v5, v8, v6);
    }
    else
    {
      v19 = v10 + v11;
      *(_DWORD *)(a1 + 8) = v19;
      if ( v16 != v5 )
      {
        v31 = v16;
        v36 = result - (_QWORD)v15;
        result = (size_t)memcpy((void *)(v12 + 8LL * v19 - v17), v5, v17);
        v16 = v31;
        v17 = v36;
      }
      if ( v18 )
      {
        for ( result = 0; result != v18; ++result )
          *(_QWORD *)&v5[8 * result] = *(_QWORD *)&v8[8 * result];
        v8 += v17;
      }
      if ( a4 != v8 )
        return (size_t)memcpy(v16, v8, a4 - v8);
    }
  }
  return result;
}
