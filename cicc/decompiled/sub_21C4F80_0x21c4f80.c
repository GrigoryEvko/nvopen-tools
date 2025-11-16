// Function: sub_21C4F80
// Address: 0x21c4f80
//
char __fastcall sub_21C4F80(
        __int64 a1,
        __int64 a2,
        __m128i a3,
        double a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        int a9,
        __int64 *a10)
{
  __int64 v13; // rdi
  unsigned __int64 v14; // r8
  int v15; // ebx
  unsigned __int64 v16; // r15
  __int64 v17; // rdx
  unsigned __int64 v19; // r10
  int v20; // r9d
  __int64 v21; // rdx
  unsigned __int64 v22; // r10
  int v23; // r9d
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rcx
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rcx
  unsigned __int64 v34; // [rsp+0h] [rbp-50h]
  unsigned __int64 v35; // [rsp+0h] [rbp-50h]
  int v36; // [rsp+Ch] [rbp-44h]
  int v37; // [rsp+Ch] [rbp-44h]
  unsigned __int64 v38; // [rsp+10h] [rbp-40h]
  unsigned __int64 v39; // [rsp+10h] [rbp-40h]
  unsigned __int64 v40; // [rsp+10h] [rbp-40h]

  v13 = (__int64)a10;
  v14 = *((unsigned int *)a10 + 2);
  v15 = *((_DWORD *)a10 + 2);
  if ( a9 == 1 )
  {
    v22 = (unsigned int)(v14 + 2);
    v23 = v14 + 2;
    if ( v22 < v14 )
    {
      *((_DWORD *)a10 + 2) = v22;
      v24 = *a10;
    }
    else if ( v22 > v14 )
    {
      v25 = *((unsigned int *)a10 + 2);
      if ( v22 > *((unsigned int *)a10 + 3) )
      {
        v34 = *((unsigned int *)a10 + 2);
        v36 = v14 + 2;
        v38 = (unsigned int)(v14 + 2);
        sub_16CD150((__int64)a10, a10 + 2, v38, 24, v14, v23);
        v13 = (__int64)a10;
        v14 = v34;
        v23 = v36;
        v22 = v38;
        v25 = *((unsigned int *)a10 + 2);
      }
      v24 = *(_QWORD *)v13;
      v26 = *(_QWORD *)v13 + 24 * v25;
      v27 = *(_QWORD *)v13 + 24 * v22;
      if ( v26 != v27 )
      {
        do
        {
          if ( v26 )
          {
            *(_QWORD *)v26 = 0;
            *(_DWORD *)(v26 + 8) = 0;
            *(_QWORD *)(v26 + 16) = 0;
          }
          v26 += 24;
        }
        while ( v27 != v26 );
        v24 = *(_QWORD *)v13;
      }
      *(_DWORD *)(v13 + 8) = v23;
    }
    else
    {
      v24 = *a10;
    }
    return sub_21C2F80(a1, a2, a7, a8, v24 + 24 * v14, v24 + 24LL * (unsigned int)(v15 + 1), a3, a4, a5);
  }
  else if ( a9 == 2 )
  {
    v16 = (unsigned int)(v14 + 1);
    if ( v16 >= v14 )
    {
      if ( v16 > v14 )
      {
        v31 = *((unsigned int *)a10 + 2);
        if ( v16 > *((unsigned int *)a10 + 3) )
        {
          v39 = *((unsigned int *)a10 + 2);
          sub_16CD150((__int64)a10, a10 + 2, (unsigned int)(v14 + 1), 24, v14, 2);
          v13 = (__int64)a10;
          v14 = v39;
          v31 = *((unsigned int *)a10 + 2);
        }
        v17 = *(_QWORD *)v13;
        v32 = *(_QWORD *)v13 + 24 * v31;
        v33 = *(_QWORD *)v13 + 24 * v16;
        if ( v32 != v33 )
        {
          do
          {
            if ( v32 )
            {
              *(_QWORD *)v32 = 0;
              *(_DWORD *)(v32 + 8) = 0;
              *(_QWORD *)(v32 + 16) = 0;
            }
            v32 += 24;
          }
          while ( v33 != v32 );
          v17 = *(_QWORD *)v13;
        }
        *(_DWORD *)(v13 + 8) = v16;
      }
      else
      {
        v17 = *a10;
      }
    }
    else
    {
      *((_DWORD *)a10 + 2) = v16;
      v17 = *a10;
    }
    return sub_21C2A00(a1, a7, a8, v17 + 24 * v14);
  }
  else
  {
    v19 = (unsigned int)(v14 + 2);
    v20 = v14 + 2;
    if ( v19 < v14 )
    {
      *((_DWORD *)a10 + 2) = v19;
      v21 = *a10;
    }
    else if ( v19 > v14 )
    {
      v28 = *((unsigned int *)a10 + 2);
      if ( v19 > *((unsigned int *)a10 + 3) )
      {
        v35 = *((unsigned int *)a10 + 2);
        v37 = v14 + 2;
        v40 = (unsigned int)(v14 + 2);
        sub_16CD150((__int64)a10, a10 + 2, v40, 24, v14, v20);
        v13 = (__int64)a10;
        v14 = v35;
        v20 = v37;
        v19 = v40;
        v28 = *((unsigned int *)a10 + 2);
      }
      v21 = *(_QWORD *)v13;
      v29 = *(_QWORD *)v13 + 24 * v28;
      v30 = *(_QWORD *)v13 + 24 * v19;
      if ( v29 != v30 )
      {
        do
        {
          if ( v29 )
          {
            *(_QWORD *)v29 = 0;
            *(_DWORD *)(v29 + 8) = 0;
            *(_QWORD *)(v29 + 16) = 0;
          }
          v29 += 24;
        }
        while ( v30 != v29 );
        v21 = *(_QWORD *)v13;
      }
      *(_DWORD *)(v13 + 8) = v20;
    }
    else
    {
      v21 = *a10;
    }
    return sub_21C2F60(a1, a2, a7, a8, v21 + 24 * v14, v21 + 24LL * (unsigned int)(v15 + 1), a3, a4, a5);
  }
}
