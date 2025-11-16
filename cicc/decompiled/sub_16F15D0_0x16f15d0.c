// Function: sub_16F15D0
// Address: 0x16f15d0
//
void __fastcall sub_16F15D0(char *a1, char *a2)
{
  int v2; // eax
  size_t v3; // r8
  char *v4; // r12
  __int64 v5; // r15
  size_t v6; // r14
  const void *v7; // rdi
  const void *v8; // rsi
  char *i; // r12
  int v10; // eax
  __int64 v11; // r13
  size_t v12; // rbx
  const void *v13; // rsi
  char *v14; // r12
  size_t v15; // [rsp+0h] [rbp-60h]
  size_t v16; // [rsp+0h] [rbp-60h]
  char *v19; // [rsp+20h] [rbp-40h]
  char *v20; // [rsp+28h] [rbp-38h]

  if ( a1 != a2 && a2 != a1 + 8 )
  {
    v19 = a1 + 8;
    v20 = a1 + 16;
    do
    {
      while ( 1 )
      {
        v5 = *(_QWORD *)v19;
        v3 = *(_QWORD *)(*(_QWORD *)a1 + 16LL);
        v6 = *(_QWORD *)(*(_QWORD *)v19 + 16LL);
        v7 = *(const void **)(*(_QWORD *)v19 + 8LL);
        v8 = *(const void **)(*(_QWORD *)a1 + 8LL);
        if ( v6 <= v3 )
          break;
        if ( !v3 )
          goto LABEL_15;
        v16 = *(_QWORD *)(*(_QWORD *)a1 + 16LL);
        v2 = memcmp(v7, v8, v16);
        v3 = v16;
        if ( v2 )
          goto LABEL_14;
LABEL_7:
        if ( v6 < v3 )
          goto LABEL_8;
LABEL_15:
        for ( i = v19; ; i -= 8 )
        {
          v11 = *((_QWORD *)i - 1);
          v12 = *(_QWORD *)(v11 + 16);
          v13 = *(const void **)(v11 + 8);
          if ( v12 < v6 )
            break;
          if ( v6 )
          {
            v10 = memcmp(v7, v13, v6);
            if ( v10 )
              goto LABEL_24;
          }
          if ( v12 == v6 )
            goto LABEL_25;
LABEL_19:
          if ( v12 <= v6 )
            goto LABEL_25;
LABEL_20:
          *(_QWORD *)i = v11;
          v7 = *(const void **)(v5 + 8);
          v6 = *(_QWORD *)(v5 + 16);
        }
        if ( !v12 )
          goto LABEL_25;
        v10 = memcmp(v7, v13, *(_QWORD *)(v11 + 16));
        if ( !v10 )
          goto LABEL_19;
LABEL_24:
        if ( v10 < 0 )
          goto LABEL_20;
LABEL_25:
        *(_QWORD *)i = v5;
        v14 = v20;
        v19 += 8;
        v20 += 8;
        if ( a2 == v14 )
          return;
      }
      if ( !v6
        || (v15 = *(_QWORD *)(*(_QWORD *)a1 + 16LL),
            v2 = memcmp(v7, v8, *(_QWORD *)(*(_QWORD *)v19 + 16LL)),
            v3 = v15,
            !v2) )
      {
        if ( v6 == v3 )
          goto LABEL_15;
        goto LABEL_7;
      }
LABEL_14:
      if ( v2 >= 0 )
        goto LABEL_15;
LABEL_8:
      v4 = v20;
      if ( a1 != v19 )
        memmove(a1 + 8, a1, v19 - a1);
      v19 += 8;
      v20 += 8;
      *(_QWORD *)a1 = v5;
    }
    while ( a2 != v4 );
  }
}
