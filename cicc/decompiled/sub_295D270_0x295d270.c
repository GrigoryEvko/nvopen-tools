// Function: sub_295D270
// Address: 0x295d270
//
char *__fastcall sub_295D270(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  char *v5; // r12
  char *v6; // r14
  __int64 v7; // rbx
  __int64 v8; // rax
  char *v9; // rbx
  __int64 v10; // rsi
  _QWORD *v11; // rax
  __int64 v12; // rdi
  _QWORD *v13; // rcx
  __int64 *v14; // rbx
  __int64 v15; // rsi
  _QWORD *v16; // rax
  _QWORD *v17; // rdx
  __int64 v19; // rsi
  char *v20; // r13
  __int64 v21; // rsi
  __int64 v22; // rsi

  v5 = (char *)a1[1];
  v6 = (char *)*a1;
  v7 = (__int64)&v5[-*a1] >> 5;
  v8 = (__int64)&v5[-*a1] >> 3;
  if ( v7 <= 0 )
  {
LABEL_41:
    if ( v8 != 2 )
    {
      if ( v8 != 3 )
      {
        if ( v8 != 1 )
        {
LABEL_44:
          v6 = v5;
          return sub_295D210((__int64)a1, v6, v5);
        }
LABEL_59:
        if ( (unsigned __int8)sub_B19060(a2, *(_QWORD *)v6, a3, a4) )
          goto LABEL_8;
        goto LABEL_44;
      }
      if ( (unsigned __int8)sub_B19060(a2, *(_QWORD *)v6, a3, a4) )
        goto LABEL_8;
      v6 += 8;
    }
    if ( (unsigned __int8)sub_B19060(a2, *(_QWORD *)v6, a3, a4) )
      goto LABEL_8;
    v6 += 8;
    goto LABEL_59;
  }
  v9 = &v6[32 * v7];
  while ( 1 )
  {
    v10 = *(_QWORD *)v6;
    if ( *(_BYTE *)(a2 + 28) )
      break;
    if ( sub_C8CA60(a2, v10) )
      goto LABEL_8;
    v19 = *((_QWORD *)v6 + 1);
    v20 = v6 + 8;
    if ( *(_BYTE *)(a2 + 28) )
    {
      v11 = *(_QWORD **)(a2 + 8);
      v12 = *(unsigned int *)(a2 + 20);
      goto LABEL_18;
    }
    if ( sub_C8CA60(a2, v19) )
      goto LABEL_22;
    v21 = *((_QWORD *)v6 + 2);
    v20 = v6 + 16;
    if ( *(_BYTE *)(a2 + 28) )
    {
      v11 = *(_QWORD **)(a2 + 8);
      a3 = *(unsigned int *)(a2 + 20);
      a4 = (__int64)&v11[a3];
LABEL_24:
      if ( (_QWORD *)a4 != v11 )
      {
        a3 = (__int64)v11;
        while ( v21 != *(_QWORD *)a3 )
        {
          a3 += 8;
          if ( a3 == a4 )
            goto LABEL_32;
        }
        goto LABEL_22;
      }
LABEL_32:
      v22 = *((_QWORD *)v6 + 3);
      v20 = v6 + 24;
      goto LABEL_33;
    }
    if ( sub_C8CA60(a2, v21) )
      goto LABEL_22;
    v22 = *((_QWORD *)v6 + 3);
    v20 = v6 + 24;
    if ( !*(_BYTE *)(a2 + 28) )
    {
      if ( sub_C8CA60(a2, v22) )
        goto LABEL_22;
      goto LABEL_39;
    }
    v11 = *(_QWORD **)(a2 + 8);
    a4 = (__int64)&v11[*(unsigned int *)(a2 + 20)];
LABEL_33:
    if ( v11 != (_QWORD *)a4 )
    {
      while ( *v11 != v22 )
      {
        if ( (_QWORD *)a4 == ++v11 )
          goto LABEL_39;
      }
LABEL_22:
      v6 = v20;
      goto LABEL_8;
    }
LABEL_39:
    v6 += 32;
    if ( v6 == v9 )
    {
      v8 = (v5 - v6) >> 3;
      goto LABEL_41;
    }
  }
  v11 = *(_QWORD **)(a2 + 8);
  v12 = *(unsigned int *)(a2 + 20);
  v13 = &v11[v12];
  a3 = (__int64)v11;
  if ( v11 == v13 )
  {
LABEL_17:
    v19 = *((_QWORD *)v6 + 1);
    v20 = v6 + 8;
LABEL_18:
    a4 = (__int64)&v11[v12];
    if ( v11 != (_QWORD *)a4 )
    {
      a3 = (__int64)v11;
      while ( *(_QWORD *)a3 != v19 )
      {
        a3 += 8;
        if ( a4 == a3 )
          goto LABEL_23;
      }
      goto LABEL_22;
    }
LABEL_23:
    v21 = *((_QWORD *)v6 + 2);
    v20 = v6 + 16;
    goto LABEL_24;
  }
  while ( v10 != *(_QWORD *)a3 )
  {
    a3 += 8;
    if ( v13 == (_QWORD *)a3 )
      goto LABEL_17;
  }
LABEL_8:
  if ( v5 != v6 )
  {
    v14 = (__int64 *)(v6 + 8);
    if ( v5 != v6 + 8 )
    {
      do
      {
        while ( 2 )
        {
          v15 = *v14;
          if ( *(_BYTE *)(a2 + 28) )
          {
            v16 = *(_QWORD **)(a2 + 8);
            v17 = &v16[*(unsigned int *)(a2 + 20)];
            if ( v16 != v17 )
            {
              while ( v15 != *v16 )
              {
                if ( v17 == ++v16 )
                  goto LABEL_53;
              }
LABEL_15:
              if ( v5 == (char *)++v14 )
                return sub_295D210((__int64)a1, v6, v5);
              continue;
            }
          }
          else
          {
            if ( sub_C8CA60(a2, v15) )
              goto LABEL_15;
            v15 = *v14;
          }
          break;
        }
LABEL_53:
        ++v14;
        *(_QWORD *)v6 = v15;
        v6 += 8;
      }
      while ( v5 != (char *)v14 );
    }
  }
  return sub_295D210((__int64)a1, v6, v5);
}
