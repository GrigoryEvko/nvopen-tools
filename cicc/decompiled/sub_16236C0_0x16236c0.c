// Function: sub_16236C0
// Address: 0x16236c0
//
__int64 __fastcall sub_16236C0(char **a1, unsigned int a2)
{
  unsigned int v2; // r13d
  char *v3; // rdx
  __int64 v4; // rax
  char *v5; // r15
  __int64 v6; // rcx
  __int64 v7; // rax
  char *v8; // rbx
  char *v9; // rax
  __int64 v10; // r12
  char *i; // r12
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rsi
  unsigned __int8 *v15; // rsi
  __int64 v16; // rsi

  v2 = a2;
  v3 = *a1;
  v4 = 16LL * *((unsigned int *)a1 + 2);
  v5 = &(*a1)[v4];
  v6 = v4 >> 4;
  v7 = v4 >> 6;
  if ( v7 )
  {
    v8 = *a1;
    v9 = &v3[64 * v7];
    while ( a2 != *(_DWORD *)v8 )
    {
      if ( a2 == *((_DWORD *)v8 + 4) )
      {
        v8 += 16;
        goto LABEL_8;
      }
      if ( a2 == *((_DWORD *)v8 + 8) )
      {
        v8 += 32;
        goto LABEL_8;
      }
      if ( a2 == *((_DWORD *)v8 + 12) )
      {
        v8 += 48;
        goto LABEL_8;
      }
      v8 += 64;
      if ( v9 == v8 )
      {
        v6 = (v5 - v8) >> 4;
        goto LABEL_30;
      }
    }
    goto LABEL_8;
  }
  v8 = *a1;
LABEL_30:
  if ( v6 == 2 )
    goto LABEL_36;
  if ( v6 == 3 )
  {
    if ( a2 == *(_DWORD *)v8 )
      goto LABEL_8;
    v8 += 16;
LABEL_36:
    if ( a2 == *(_DWORD *)v8 )
      goto LABEL_8;
    v8 += 16;
    goto LABEL_38;
  }
  if ( v6 != 1 )
    goto LABEL_33;
LABEL_38:
  if ( a2 != *(_DWORD *)v8 )
    goto LABEL_33;
LABEL_8:
  if ( v5 == v8 )
  {
LABEL_33:
    v8 = v5;
    v2 = 0;
    goto LABEL_25;
  }
  v10 = (__int64)(v8 + 16);
  if ( v5 == v8 + 16 )
  {
    v2 = 1;
    do
    {
LABEL_21:
      v16 = *(_QWORD *)(v10 - 8);
      v10 -= 16;
      if ( v16 )
        sub_161E7C0(v10 + 8, v16);
    }
    while ( (char *)v10 != v8 );
    v3 = *a1;
    goto LABEL_25;
  }
  for ( i = v8 + 24; ; i += 16 )
  {
    v12 = *((_DWORD *)i - 2);
    if ( v2 != v12 )
    {
      v13 = (__int64)(v8 + 8);
      *(_DWORD *)v8 = v12;
      if ( v8 + 8 != i )
      {
        v14 = *((_QWORD *)v8 + 1);
        if ( v14 )
        {
          sub_161E7C0((__int64)(v8 + 8), v14);
          v13 = (__int64)(v8 + 8);
        }
        v15 = *(unsigned __int8 **)i;
        *((_QWORD *)v8 + 1) = *(_QWORD *)i;
        if ( v15 )
        {
          sub_1623210((__int64)i, v15, v13);
          *(_QWORD *)i = 0;
        }
      }
      v8 += 16;
    }
    if ( v5 == i + 8 )
      break;
  }
  v3 = *a1;
  v10 = (__int64)&(*a1)[16 * *((unsigned int *)a1 + 2)];
  LOBYTE(v2) = v8 != (char *)v10;
  if ( v8 != (char *)v10 )
    goto LABEL_21;
LABEL_25:
  *((_DWORD *)a1 + 2) = (v8 - v3) >> 4;
  return v2;
}
