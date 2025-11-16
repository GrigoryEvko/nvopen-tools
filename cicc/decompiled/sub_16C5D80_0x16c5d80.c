// Function: sub_16C5D80
// Address: 0x16c5d80
//
void __fastcall sub_16C5D80(__int64 *a1, __int64 *a2)
{
  __int64 v3; // rax
  int v4; // edx
  int v5; // eax
  int v6; // edx
  int v7; // eax
  __int64 v8; // rdx
  unsigned __int64 v9; // r14
  int v10; // eax
  unsigned __int64 v11; // r15
  unsigned __int64 v12; // r13
  __int64 v13; // rax
  char *v14; // rdx
  char *v15; // rcx
  char v16; // si
  const void *v17; // rsi
  int v18; // edx
  const void *v19; // rsi

  if ( a1 == a2 )
    return;
  v3 = *a1;
  if ( a1 + 2 != (__int64 *)*a1 && (__int64 *)*a2 != a2 + 2 )
  {
    *a1 = *a2;
    v4 = *((_DWORD *)a2 + 2);
    *a2 = v3;
    v5 = *((_DWORD *)a1 + 2);
    *((_DWORD *)a1 + 2) = v4;
    v6 = *((_DWORD *)a2 + 3);
    *((_DWORD *)a2 + 2) = v5;
    v7 = *((_DWORD *)a1 + 3);
    *((_DWORD *)a1 + 3) = v6;
    *((_DWORD *)a2 + 3) = v7;
    return;
  }
  v8 = *((unsigned int *)a2 + 2);
  if ( *((_DWORD *)a1 + 3) < (unsigned int)v8 )
  {
    sub_16CD150(a1, a1 + 2, v8, 1);
    v9 = *((unsigned int *)a1 + 2);
    v10 = v9;
    if ( *((_DWORD *)a2 + 3) >= (unsigned int)v9 )
      goto LABEL_8;
    goto LABEL_22;
  }
  v9 = *((unsigned int *)a1 + 2);
  v10 = v9;
  if ( *((_DWORD *)a2 + 3) < (unsigned int)v9 )
  {
LABEL_22:
    sub_16CD150(a2, a2 + 2, v9, 1);
    v9 = *((unsigned int *)a1 + 2);
    v10 = *((_DWORD *)a1 + 2);
  }
LABEL_8:
  v11 = *((unsigned int *)a2 + 2);
  v12 = v9;
  if ( v11 <= v9 )
    v12 = *((unsigned int *)a2 + 2);
  if ( v12 )
  {
    v13 = 0;
    do
    {
      v14 = (char *)(v13 + *a2);
      v15 = (char *)(v13 + *a1);
      ++v13;
      v16 = *v15;
      *v15 = *v14;
      *v14 = v16;
    }
    while ( v12 != v13 );
    v9 = *((unsigned int *)a1 + 2);
    v11 = *((unsigned int *)a2 + 2);
    v10 = *((_DWORD *)a1 + 2);
  }
  if ( v11 >= v9 )
  {
    if ( v11 > v9 )
    {
      v18 = v9;
      v19 = (const void *)(*a2 + v12);
      if ( v19 != (const void *)(v11 + *a2) )
      {
        memcpy((void *)(v9 + *a1), v19, v11 - v12);
        v18 = *((_DWORD *)a1 + 2);
      }
      *((_DWORD *)a1 + 2) = v18 + v11 - v9;
      *((_DWORD *)a2 + 2) = v12;
    }
  }
  else
  {
    v17 = (const void *)(*a1 + v12);
    if ( v17 != (const void *)(v9 + *a1) )
    {
      memcpy((void *)(v11 + *a2), v17, v9 - v12);
      v10 = v9 + *((_DWORD *)a2 + 2) - v11;
    }
    *((_DWORD *)a2 + 2) = v10;
    *((_DWORD *)a1 + 2) = v12;
  }
}
