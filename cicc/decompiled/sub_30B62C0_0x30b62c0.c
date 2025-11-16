// Function: sub_30B62C0
// Address: 0x30b62c0
//
char __fastcall sub_30B62C0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  __int64 v7; // r9
  __int64 *v8; // r12
  __int64 v9; // rax
  char *v10; // rbx
  char result; // al
  __int64 v12; // rbx
  __int64 i; // r8
  __int64 v14; // r12
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 v17; // rax
  char *v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rax
  char *v21; // rdi
  char *v22; // rax
  char *v23; // rsi
  __int64 v24; // rax
  char *v25; // r14
  __int64 v26; // rax
  __int64 v27; // rax
  char *v30; // [rsp+18h] [rbp-68h]
  __int64 v31; // [rsp+18h] [rbp-68h]
  __int64 v32; // [rsp+28h] [rbp-58h] BYREF
  _BYTE *v33; // [rsp+30h] [rbp-50h] BYREF
  __int64 v34; // [rsp+38h] [rbp-48h]
  _BYTE v35[64]; // [rsp+40h] [rbp-40h] BYREF

  v6 = *(unsigned int *)(a2 + 8);
  v7 = *(_QWORD *)a2;
  v8 = *(__int64 **)(*(_QWORD *)a2 + 8LL * ((int)v6 - 1));
  if ( (_DWORD)v6 != 1 )
  {
    v9 = 8 * v6;
    v30 = (char *)(v7 + v9);
    if ( v7 != v7 + v9 )
    {
      v10 = *(char **)a2;
      while ( 1 )
      {
        sub_310BF50(a1, *(_QWORD *)v10, v8, &v32, &v33);
        result = sub_D968A0((__int64)v33);
        if ( !result )
          return result;
        v10 += 8;
        *((_QWORD *)v10 - 1) = v32;
        if ( v30 == v10 )
        {
          v30 = *(char **)a2;
          v9 = 8LL * *(unsigned int *)(a2 + 8);
          v18 = (char *)(*(_QWORD *)a2 + v9);
          goto LABEL_21;
        }
      }
    }
    v18 = (char *)(v7 + v9);
LABEL_21:
    v19 = v9 >> 3;
    v20 = v9 >> 5;
    if ( v20 )
    {
      v21 = v30;
      v22 = &v30[32 * v20];
      while ( *(_WORD *)(*(_QWORD *)v21 + 24LL) )
      {
        if ( !*(_WORD *)(*((_QWORD *)v21 + 1) + 24LL) )
        {
          v21 += 8;
          goto LABEL_28;
        }
        if ( !*(_WORD *)(*((_QWORD *)v21 + 2) + 24LL) )
        {
          v21 += 16;
          goto LABEL_28;
        }
        if ( !*(_WORD *)(*((_QWORD *)v21 + 3) + 24LL) )
        {
          v21 += 24;
          goto LABEL_28;
        }
        v21 += 32;
        if ( v22 == v21 )
        {
          v19 = (v18 - v21) >> 3;
          goto LABEL_45;
        }
      }
      goto LABEL_28;
    }
    v21 = v30;
LABEL_45:
    if ( v19 != 2 )
    {
      if ( v19 != 3 )
      {
        v25 = v18;
        if ( v19 != 1 )
          goto LABEL_35;
        goto LABEL_48;
      }
      if ( !*(_WORD *)(*(_QWORD *)v21 + 24LL) )
      {
LABEL_28:
        if ( v21 == v18 || (v23 = v21 + 8, v21 + 8 == v18) )
        {
          v25 = v21;
        }
        else
        {
          do
          {
            if ( *(_WORD *)(*(_QWORD *)v23 + 24LL) )
            {
              *(_QWORD *)v21 = *(_QWORD *)v23;
              v21 += 8;
            }
            v23 += 8;
          }
          while ( v23 != v18 );
          v24 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
          v30 = *(char **)a2;
          v25 = &v21[v24 - (_QWORD)v23];
          if ( v23 != (char *)v24 )
          {
            memmove(v21, v23, v24 - (_QWORD)v23);
            v30 = *(char **)a2;
          }
        }
        goto LABEL_35;
      }
      v21 += 8;
    }
    if ( *(_WORD *)(*(_QWORD *)v21 + 24LL) )
    {
      v21 += 8;
LABEL_48:
      v25 = v18;
      if ( !*(_WORD *)(*(_QWORD *)v21 + 24LL) )
        goto LABEL_28;
LABEL_35:
      v26 = (v25 - v30) >> 3;
      *(_DWORD *)(a2 + 8) = v26;
      if ( !(_DWORD)v26 || (result = sub_30B62C0(a1, a2, a3)) != 0 )
      {
        v27 = *(unsigned int *)(a3 + 8);
        if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
        {
          sub_C8D5F0(a3, (const void *)(a3 + 16), v27 + 1, 8u, a5, v7);
          v27 = *(unsigned int *)(a3 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a3 + 8 * v27) = v8;
        ++*(_DWORD *)(a3 + 8);
        return 1;
      }
      return result;
    }
    goto LABEL_28;
  }
  if ( *((_WORD *)v8 + 12) == 6 )
  {
    v33 = v35;
    v34 = 0x200000000LL;
    v12 = v8[4];
    for ( i = v12 + 8 * v8[5]; i != v12; LODWORD(v34) = v34 + 1 )
    {
      while ( 1 )
      {
        v14 = *(_QWORD *)v12;
        if ( *(_WORD *)(*(_QWORD *)v12 + 24LL) )
          break;
        v12 += 8;
        if ( i == v12 )
          goto LABEL_15;
      }
      v15 = (unsigned int)v34;
      v16 = (unsigned int)v34 + 1LL;
      if ( v16 > HIDWORD(v34) )
      {
        v31 = i;
        sub_C8D5F0((__int64)&v33, v35, v16, 8u, i, v7);
        v15 = (unsigned int)v34;
        i = v31;
      }
      v12 += 8;
      *(_QWORD *)&v33[8 * v15] = v14;
    }
LABEL_15:
    v8 = sub_DC8BD0(a1, (__int64)&v33, 0, 0);
    if ( v33 != v35 )
      _libc_free((unsigned __int64)v33);
  }
  v17 = *(unsigned int *)(a3 + 8);
  if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, (const void *)(a3 + 16), v17 + 1, 8u, a5, v7);
    v17 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v17) = v8;
  ++*(_DWORD *)(a3 + 8);
  return 1;
}
