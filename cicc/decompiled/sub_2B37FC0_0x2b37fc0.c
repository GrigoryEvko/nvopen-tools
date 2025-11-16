// Function: sub_2B37FC0
// Address: 0x2b37fc0
//
unsigned __int64 __fastcall sub_2B37FC0(__int64 a1, __int64 a2, char *a3, _QWORD *a4, __int64 a5, __int64 a6)
{
  unsigned __int64 result; // rax
  __int64 v10; // rcx
  unsigned int v11; // esi
  int v12; // eax
  __int64 *v13; // rsi
  unsigned int v14; // r13d
  unsigned __int64 v15; // rdx
  unsigned int v16; // eax
  signed __int64 v17; // r13
  unsigned __int64 v18; // rdx
  __int64 v19; // rdi
  int v20; // ecx
  unsigned __int64 v21; // r14
  __int64 v22; // rax
  unsigned __int64 v23; // r14
  __int64 v24; // rsi
  __int64 v25; // rsi
  __int64 v26; // rdx
  int v27; // ecx
  _DWORD *v28; // rdi
  signed __int64 v29; // rax
  int v30; // edx
  unsigned __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rsi
  __int64 i; // rdx
  _DWORD *v35; // rcx
  bool v36; // cc

  result = (unsigned __int64)a4;
  v10 = *(unsigned int *)(a1 + 88);
  if ( !(_DWORD)v10 )
  {
    v17 = 4 * result;
    v18 = *(unsigned int *)(a1 + 28);
    *(_DWORD *)(a1 + 24) = 0;
    LODWORD(result) = 0;
    v19 = 0;
    if ( v17 >> 2 > v18 )
    {
      sub_C8D5F0(a1 + 16, (const void *)(a1 + 32), v17 >> 2, 4u, a5, a6);
      result = *(unsigned int *)(a1 + 24);
      v19 = 4 * result;
    }
    if ( v17 )
    {
      memcpy((void *)(*(_QWORD *)(a1 + 16) + v19), a3, v17);
      LODWORD(result) = *(_DWORD *)(a1 + 24);
    }
    v20 = *(_DWORD *)(a1 + 92);
    v21 = a2 & 0xFFFFFFFFFFFFFFFBLL;
    *(_DWORD *)(a1 + 24) = result + (v17 >> 2);
    if ( v20 )
    {
      if ( *(_DWORD *)(a1 + 88) )
      {
        **(_QWORD **)(a1 + 80) = v21;
        result = *(unsigned int *)(a1 + 88);
        if ( (_DWORD)result )
        {
LABEL_21:
          *(_DWORD *)(a1 + 88) = 1;
          return result;
        }
      }
    }
    else
    {
      *(_DWORD *)(a1 + 88) = 0;
      sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), 1u, 8u, a5, a6);
    }
    result = *(_QWORD *)(a1 + 80);
    if ( result )
      *(_QWORD *)result = v21;
    goto LABEL_21;
  }
  if ( !(_BYTE)a5 )
  {
    v11 = 1;
    if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 17 )
      v11 = *(_DWORD *)(*(_QWORD *)a1 + 32LL);
    v12 = *(_DWORD *)(*(_QWORD *)(a2 + 8) + 32LL) / v11;
    v13 = *(__int64 **)(a1 + 80);
    v14 = v12;
    if ( (_DWORD)v10 == 2 )
    {
      v29 = sub_2B35AF0(a1, v13, (__int64)(v13 + 1), *(const void **)(a1 + 16), *(unsigned int *)(a1 + 24), a6);
      if ( v30 == 1 )
        *(_DWORD *)(a1 + 128) = 1;
      if ( __OFADD__(*(_QWORD *)(a1 + 120), v29) )
      {
        v36 = v29 <= 0;
        v31 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v36 )
          v31 = 0x8000000000000000LL;
      }
      else
      {
        v31 = *(_QWORD *)(a1 + 120) + v29;
      }
      *(_QWORD *)(a1 + 120) = v31;
      v32 = *(unsigned int *)(a1 + 24);
      v33 = *(_QWORD *)(a1 + 16);
      if ( (_DWORD)v32 )
      {
        for ( i = 0; i != v32; ++i )
        {
          v35 = (_DWORD *)(v33 + 4LL * (unsigned int)i);
          if ( *v35 != -1 )
            *v35 = i;
        }
        LODWORD(v32) = *(_DWORD *)(a1 + 24);
      }
      v10 = *(unsigned int *)(a1 + 88);
      if ( v14 < (unsigned int)v32 )
        v14 = v32;
    }
    else
    {
      v15 = *v13 & 0xFFFFFFFFFFFFFFF8LL;
      if ( *v13 && (*v13 & 4) != 0 && v15 )
      {
        v16 = *(_DWORD *)(v15 + 120);
        if ( !v16 )
          v16 = *(_DWORD *)(v15 + 8);
        if ( v14 < v16 )
          v14 = v16;
      }
      else
      {
        v22 = *(_QWORD *)(v15 + 8);
        if ( *(_DWORD *)(v22 + 32) >= v14 )
          v14 = *(_DWORD *)(v22 + 32);
      }
    }
    v23 = a2 & 0xFFFFFFFFFFFFFFFBLL;
    if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 92) )
    {
      sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), v10 + 1, 8u, a5, a6);
      v10 = *(unsigned int *)(a1 + 88);
    }
    result = *(_QWORD *)(a1 + 80);
    *(_QWORD *)(result + 8 * v10) = v23;
    v24 = *(unsigned int *)(a1 + 24);
    ++*(_DWORD *)(a1 + 88);
    if ( (_DWORD)v24 )
    {
      v25 = 4 * v24;
      v26 = 0;
      do
      {
        v27 = *(_DWORD *)&a3[v26];
        if ( v27 != -1 )
        {
          v28 = (_DWORD *)(v26 + *(_QWORD *)(a1 + 16));
          if ( *v28 == -1 )
            *v28 = v14 + v27;
        }
        v26 += 4;
      }
      while ( v25 != v26 );
    }
  }
  return result;
}
