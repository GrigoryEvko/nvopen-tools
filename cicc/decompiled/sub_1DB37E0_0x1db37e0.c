// Function: sub_1DB37E0
// Address: 0x1db37e0
//
unsigned __int64 __fastcall sub_1DB37E0(__int64 **a1, _QWORD *a2, __int64 a3)
{
  __int64 *v5; // r8
  __int64 v7; // r11
  __int64 *v8; // rsi
  unsigned int v9; // edi
  __int64 *v10; // rdx
  __int64 v11; // rdx
  __int64 *v12; // rbx
  __int64 v13; // rax
  __int64 *v14; // rcx
  size_t v15; // r12
  char *v16; // r8
  __int64 v17; // r8
  __int64 v19; // rax

  v5 = a2 + 3;
  v7 = a2[2];
  v8 = (__int64 *)(**a1 + 24LL * *((unsigned int *)*a1 + 2));
  v9 = *(_DWORD *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a3 >> 1) & 3;
  if ( v5 != v8 )
  {
    v10 = v5;
    while ( (*(_DWORD *)((v10[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v10[1] >> 1) & 3) <= v9 )
    {
      v10 += 3;
      if ( v10 == v8 )
        goto LABEL_6;
    }
    v8 = v10;
  }
LABEL_6:
  v11 = *(v8 - 2);
  if ( (*(_DWORD *)((v11 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v11 >> 1) & 3) > v9 )
    a2[1] = v11;
  else
    a2[1] = a3;
  v12 = *a1;
  v13 = **a1;
  v14 = (__int64 *)(v13 + 24LL * *((unsigned int *)*a1 + 2));
  if ( v14 != v8 )
  {
    if ( (*(_DWORD *)((*v8 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v8 >> 1) & 3) <= (*(_DWORD *)((a2[1] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                           | (unsigned int)((__int64)a2[1] >> 1)
                                                                                           & 3)
      && v8[2] == v7 )
    {
      v19 = v8[1];
      v8 += 3;
      a2[1] = v19;
      v12 = *a1;
      v13 = **a1;
      v15 = v13 + 24LL * *((unsigned int *)*a1 + 2) - (_QWORD)v8;
      if ( v8 == (__int64 *)(v13 + 24LL * *((unsigned int *)*a1 + 2)) )
      {
        v5 = (__int64 *)((char *)v5 + v15);
        goto LABEL_12;
      }
    }
    else
    {
      v15 = (char *)v14 - (char *)v8;
    }
    v16 = (char *)memmove(v5, v8, v15);
    v13 = *v12;
    v5 = (__int64 *)&v16[v15];
  }
LABEL_12:
  v17 = (__int64)v5 - v13;
  *((_DWORD *)v12 + 2) = -1431655765 * (v17 >> 3);
  return 0xAAAAAAAAAAAAAAABLL;
}
