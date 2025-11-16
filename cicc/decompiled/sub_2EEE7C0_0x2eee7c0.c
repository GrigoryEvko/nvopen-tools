// Function: sub_2EEE7C0
// Address: 0x2eee7c0
//
unsigned __int64 __fastcall sub_2EEE7C0(
        __int64 a1,
        char *a2,
        unsigned __int16 a3,
        __int16 *a4,
        unsigned __int16 a5,
        __int64 a6)
{
  char *v6; // r10
  __int16 *v7; // r15
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // r8
  char *v13; // rax
  __int64 v14; // rsi
  unsigned __int64 result; // rax
  char *v16; // r14
  __int16 *v17; // rax
  unsigned __int64 v18; // rdx
  bool v19; // zf
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rdx
  size_t v22; // rdx
  unsigned __int64 v23; // r9
  unsigned int v24; // eax
  __int16 v25; // dx
  __int64 v26; // r9
  __int16 v27; // dx
  int v28; // r9d
  char *v29; // rdx
  __int64 v30; // r9
  __int64 v31; // rsi
  _DWORD *v32; // rcx
  char *v33; // rsi
  size_t v34; // rdx
  char *v35; // rdi
  char *v36; // [rsp+0h] [rbp-60h]
  unsigned __int64 v37; // [rsp+8h] [rbp-58h]
  __int64 v38; // [rsp+10h] [rbp-50h]
  unsigned __int64 v39; // [rsp+18h] [rbp-48h]
  unsigned __int64 v40; // [rsp+20h] [rbp-40h]
  __int64 v41; // [rsp+20h] [rbp-40h]
  int v42; // [rsp+20h] [rbp-40h]
  int v43; // [rsp+20h] [rbp-40h]
  char *v44; // [rsp+28h] [rbp-38h]
  char *v45; // [rsp+28h] [rbp-38h]
  char *v46; // [rsp+28h] [rbp-38h]

  v6 = a2;
  v7 = (__int16 *)a6;
  v11 = *(_QWORD *)a1;
  v12 = *(unsigned int *)(a1 + 8);
  v13 = &a2[-*(_QWORD *)a1];
  v14 = 4 * v12;
  v44 = v13;
  result = *(unsigned int *)(a1 + 12);
  v16 = (char *)(*(_QWORD *)a1 + 4 * v12);
  v40 = result;
  if ( v6 == v16 )
  {
    if ( a4 == (__int16 *)a6 )
    {
      v28 = *(_DWORD *)(a1 + 8);
      if ( v12 > result )
      {
        result = sub_C8D5F0(a1, (const void *)(a1 + 16), v12, 4u, v12, v12);
        v28 = *(_DWORD *)(a1 + 8);
      }
    }
    else
    {
      result = (unsigned __int64)a4;
      v26 = 0;
      do
      {
        v19 = *(_WORD *)result == 0;
        result += 2LL;
        if ( v19 )
          result = 0;
        ++v26;
      }
      while ( v7 != (__int16 *)result );
      if ( v12 + v26 > v40 )
      {
        v42 = v26;
        sub_C8D5F0(a1, (const void *)(a1 + 16), v12 + v26, 4u, v12, v26);
        LODWORD(v26) = v42;
        result = *(_QWORD *)a1;
        v16 = (char *)(*(_QWORD *)a1 + 4LL * *(unsigned int *)(a1 + 8));
      }
      do
      {
        if ( v16 )
          *(_DWORD *)v16 = a5;
        v27 = *a4++;
        a3 += v27;
        if ( !v27 )
          a4 = 0;
        a5 = a3;
        v16 += 4;
      }
      while ( v7 != a4 );
      v28 = *(_DWORD *)(a1 + 8) + v26;
    }
    *(_DWORD *)(a1 + 8) = v28;
  }
  else
  {
    if ( a4 == (__int16 *)a6 )
    {
      v21 = *(unsigned int *)(a1 + 8);
      v20 = 0;
    }
    else
    {
      v17 = a4;
      v18 = 0;
      do
      {
        a6 = (__int64)(v17 + 1);
        v19 = *v17++ == 0;
        if ( v19 )
          v17 = 0;
        ++v18;
      }
      while ( v17 != v7 );
      v20 = v18;
      v21 = v12 + v18;
    }
    if ( v21 > v40 )
    {
      v39 = v20;
      sub_C8D5F0(a1, (const void *)(a1 + 16), v21, 4u, v12, a6);
      v12 = *(unsigned int *)(a1 + 8);
      v11 = *(_QWORD *)a1;
      v14 = 4 * v12;
      v6 = &v44[*(_QWORD *)a1];
      v20 = v39;
      v16 = (char *)(*(_QWORD *)a1 + 4 * v12);
    }
    v22 = v14 - (_QWORD)v44;
    v23 = (v14 - (__int64)v44) >> 2;
    if ( v23 >= v20 )
    {
      v29 = v16;
      v30 = 4 * (v12 - v20);
      result = v11 + v30;
      v31 = (v14 - v30) >> 2;
      v43 = v31;
      if ( v31 + v12 > *(unsigned int *)(a1 + 12) )
      {
        v36 = v6;
        v37 = v11 + v30;
        v38 = v30;
        sub_C8D5F0(a1, (const void *)(a1 + 16), v31 + v12, 4u, v12, v30);
        v6 = v36;
        result = v37;
        v30 = v38;
        v12 = *(unsigned int *)(a1 + 8);
        v29 = (char *)(*(_QWORD *)a1 + 4 * v12);
      }
      if ( (char *)result != v16 )
      {
        v32 = (_DWORD *)result;
        v33 = &v16[(_QWORD)v29 - result];
        do
        {
          if ( v29 )
            *(_DWORD *)v29 = *v32;
          v29 += 4;
          ++v32;
        }
        while ( v29 != v33 );
        LODWORD(v12) = *(_DWORD *)(a1 + 8);
      }
      *(_DWORD *)(a1 + 8) = v43 + v12;
      if ( (char *)result != v6 )
      {
        v34 = v30 - (_QWORD)v44;
        v35 = &v16[-(v30 - (_QWORD)v44)];
        v46 = v6;
        result = (unsigned __int64)memmove(v35, v6, v34);
        v6 = v46;
      }
      for ( ; a4 != v7; a5 = a3 )
      {
        ++a4;
        v6 += 4;
        *((_DWORD *)v6 - 1) = a5;
        a3 += *(a4 - 1);
        if ( !*(a4 - 1) )
          a4 = 0;
      }
    }
    else
    {
      v24 = v12 + v20;
      *(_DWORD *)(a1 + 8) = v24;
      if ( v6 != v16 )
      {
        v41 = (v14 - (__int64)v44) >> 2;
        v45 = v6;
        memcpy((void *)(v11 + 4LL * v24 - v22), v6, v22);
        v23 = v41;
        v6 = v45;
      }
      result = 0;
      if ( v23 )
      {
        do
        {
          ++a4;
          *(_DWORD *)&v6[4 * result] = a5;
          a3 += *(a4 - 1);
          if ( !*(a4 - 1) )
            a4 = 0;
          ++result;
          a5 = a3;
        }
        while ( v23 != result );
      }
      for ( ; a4 != v7; v16 += 4 )
      {
        if ( v16 )
          *(_DWORD *)v16 = a5;
        v25 = *a4++;
        a3 += v25;
        if ( !v25 )
          a4 = 0;
        a5 = a3;
      }
    }
  }
  return result;
}
