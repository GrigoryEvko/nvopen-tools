// Function: sub_3061580
// Address: 0x3061580
//
__int64 __fastcall sub_3061580(unsigned int *a1, __int64 a2, __int64 a3, unsigned __int64 *a4)
{
  _QWORD *v6; // r13
  _QWORD *v7; // r12
  char *v8; // rsi
  _QWORD *v9; // r13
  char *v10; // rsi
  _QWORD *v11; // rdx
  _QWORD *v12; // rbx
  _QWORD *v13; // r13
  char *v14; // rsi
  _QWORD *v15; // rdx
  _QWORD *v16; // rbx
  _QWORD *v17; // r13
  char *v18; // rsi
  _QWORD *v19; // rdx
  _QWORD *v20; // rbx
  _QWORD *v21; // r13
  char *v22; // rsi
  _QWORD *v23; // rbx
  _QWORD *v24; // rbx
  _QWORD *v25; // r13
  char *v26; // rsi
  _QWORD *v27; // rbx
  _QWORD *v28; // r13
  char *v29; // rsi
  _QWORD *v30; // rbx
  _QWORD *v31; // [rsp+0h] [rbp-50h] BYREF
  _QWORD *v32; // [rsp+8h] [rbp-48h]

  if ( a3 != 9 )
  {
    if ( a3 == 13 )
    {
      if ( *(_QWORD *)a2 == 0x664F3C74706F766ELL && *(_DWORD *)(a2 + 8) == 2019650915 && *(_BYTE *)(a2 + 12) == 62 )
      {
        sub_3150D70(&v31, *((_QWORD *)a1 + 1) + 539448LL, 2);
        v25 = v31;
        v7 = v32;
        if ( v31 == v32 )
          goto LABEL_35;
        do
        {
          v26 = (char *)a4[1];
          if ( v26 == (char *)a4[2] )
          {
            sub_2275C60(a4, v26, v25);
          }
          else
          {
            if ( v26 )
            {
              *(_QWORD *)v26 = *v25;
              *v25 = 0;
              v26 = (char *)a4[1];
            }
            a4[1] = (unsigned __int64)(v26 + 8);
          }
          ++v25;
        }
        while ( v7 != v25 );
        v27 = v32;
        v7 = v31;
        if ( v32 == v31 )
          goto LABEL_35;
        do
        {
          if ( *v7 )
            (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v7 + 8LL))(*v7);
          ++v7;
        }
        while ( v27 != v7 );
        goto LABEL_34;
      }
      if ( *(_QWORD *)a2 == 0x664F3C74706F766ELL && *(_DWORD *)(a2 + 8) == 1684630883 && *(_BYTE *)(a2 + 12) == 62 )
      {
        sub_3150D70(&v31, *((_QWORD *)a1 + 1) + 539448LL, 1);
        v28 = v31;
        v7 = v32;
        if ( v31 == v32 )
          goto LABEL_35;
        do
        {
          v29 = (char *)a4[1];
          if ( v29 == (char *)a4[2] )
          {
            sub_2275C60(a4, v29, v28);
          }
          else
          {
            if ( v29 )
            {
              *(_QWORD *)v29 = *v28;
              *v28 = 0;
              v29 = (char *)a4[1];
            }
            a4[1] = (unsigned __int64)(v29 + 8);
          }
          ++v28;
        }
        while ( v7 != v28 );
        v30 = v32;
        v7 = v31;
        if ( v32 == v31 )
          goto LABEL_35;
        do
        {
          if ( *v7 )
            (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v7 + 8LL))(*v7);
          ++v7;
        }
        while ( v30 != v7 );
        goto LABEL_34;
      }
      if ( *(_QWORD *)a2 == 0x664F3C74706F766ELL && *(_DWORD *)(a2 + 8) == 1852403043 && *(_BYTE *)(a2 + 12) == 62 )
      {
        sub_3150D70(&v31, *((_QWORD *)a1 + 1) + 539448LL, 0);
        v6 = v31;
        v7 = v32;
        if ( v31 == v32 )
          goto LABEL_35;
        do
        {
          v8 = (char *)a4[1];
          if ( v8 == (char *)a4[2] )
          {
            sub_2275C60(a4, v8, v6);
          }
          else
          {
            if ( v8 )
            {
              *(_QWORD *)v8 = *v6;
              *v6 = 0;
              v8 = (char *)a4[1];
            }
            a4[1] = (unsigned __int64)(v8 + 8);
          }
          ++v6;
        }
        while ( v7 != v6 );
        v24 = v32;
        v7 = v31;
        if ( v32 == v31 )
          goto LABEL_35;
        do
        {
          if ( *v7 )
            (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v7 + 8LL))(*v7);
          ++v7;
        }
        while ( v24 != v7 );
        goto LABEL_34;
      }
    }
    return 0;
  }
  if ( *(_QWORD *)a2 == 0x304F3C74706F766ELL && *(_BYTE *)(a2 + 8) == 62 )
  {
    sub_31504F0(&v31, 0, *((_QWORD *)a1 + 1) + 539448LL, *a1, *((unsigned __int8 *)a1 + 4));
    v9 = v31;
    v7 = v32;
    if ( v31 == v32 )
      goto LABEL_35;
    do
    {
      while ( 1 )
      {
        v10 = (char *)a4[1];
        if ( v10 != (char *)a4[2] )
          break;
        v11 = v9++;
        sub_2275C60(a4, v10, v11);
        if ( v7 == v9 )
          goto LABEL_30;
      }
      if ( v10 )
      {
        *(_QWORD *)v10 = *v9;
        *v9 = 0;
        v10 = (char *)a4[1];
      }
      ++v9;
      a4[1] = (unsigned __int64)(v10 + 8);
    }
    while ( v7 != v9 );
LABEL_30:
    v12 = v32;
    v7 = v31;
    if ( v32 == v31 )
      goto LABEL_35;
    do
    {
      if ( *v7 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v7 + 8LL))(*v7);
      ++v7;
    }
    while ( v12 != v7 );
    goto LABEL_34;
  }
  if ( *(_QWORD *)a2 == 0x314F3C74706F766ELL && *(_BYTE *)(a2 + 8) == 62 )
  {
    sub_31504F0(&v31, 1, *((_QWORD *)a1 + 1) + 539448LL, *a1, *((unsigned __int8 *)a1 + 4));
    v13 = v31;
    v7 = v32;
    if ( v31 != v32 )
    {
      do
      {
        while ( 1 )
        {
          v14 = (char *)a4[1];
          if ( v14 != (char *)a4[2] )
            break;
          v15 = v13++;
          sub_2275C60(a4, v14, v15);
          if ( v7 == v13 )
            goto LABEL_46;
        }
        if ( v14 )
        {
          *(_QWORD *)v14 = *v13;
          *v13 = 0;
          v14 = (char *)a4[1];
        }
        ++v13;
        a4[1] = (unsigned __int64)(v14 + 8);
      }
      while ( v7 != v13 );
LABEL_46:
      v16 = v32;
      v7 = v31;
      if ( v32 != v31 )
      {
        do
        {
          if ( *v7 )
            (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v7 + 8LL))(*v7);
          ++v7;
        }
        while ( v16 != v7 );
        v7 = v31;
      }
    }
    goto LABEL_35;
  }
  if ( *(_QWORD *)a2 == 0x324F3C74706F766ELL && *(_BYTE *)(a2 + 8) == 62 )
  {
    sub_31504F0(&v31, 2, *((_QWORD *)a1 + 1) + 539448LL, *a1, *((unsigned __int8 *)a1 + 4));
    v17 = v31;
    v7 = v32;
    if ( v31 != v32 )
    {
      do
      {
        while ( 1 )
        {
          v18 = (char *)a4[1];
          if ( v18 != (char *)a4[2] )
            break;
          v19 = v17++;
          sub_2275C60(a4, v18, v19);
          if ( v7 == v17 )
            goto LABEL_59;
        }
        if ( v18 )
        {
          *(_QWORD *)v18 = *v17;
          *v17 = 0;
          v18 = (char *)a4[1];
        }
        ++v17;
        a4[1] = (unsigned __int64)(v18 + 8);
      }
      while ( v7 != v17 );
LABEL_59:
      v20 = v32;
      v7 = v31;
      if ( v32 != v31 )
      {
        do
        {
          if ( *v7 )
            (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v7 + 8LL))(*v7);
          ++v7;
        }
        while ( v20 != v7 );
        v7 = v31;
      }
    }
    goto LABEL_35;
  }
  if ( *(_QWORD *)a2 != 0x334F3C74706F766ELL || *(_BYTE *)(a2 + 8) != 62 )
    return 0;
  sub_31504F0(&v31, 3, *((_QWORD *)a1 + 1) + 539448LL, *a1, *((unsigned __int8 *)a1 + 4));
  v21 = v31;
  v7 = v32;
  if ( v31 == v32 )
    goto LABEL_35;
  do
  {
    v22 = (char *)a4[1];
    if ( v22 == (char *)a4[2] )
    {
      sub_2275C60(a4, v22, v21);
    }
    else
    {
      if ( v22 )
      {
        *(_QWORD *)v22 = *v21;
        *v21 = 0;
        v22 = (char *)a4[1];
      }
      a4[1] = (unsigned __int64)(v22 + 8);
    }
    ++v21;
  }
  while ( v7 != v21 );
  v23 = v32;
  v7 = v31;
  if ( v32 == v31 )
    goto LABEL_35;
  do
  {
    if ( *v7 )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v7 + 8LL))(*v7);
    ++v7;
  }
  while ( v23 != v7 );
LABEL_34:
  v7 = v31;
LABEL_35:
  if ( v7 )
    j_j___libc_free_0((unsigned __int64)v7);
  return 1;
}
