// Function: sub_C68ED0
// Address: 0xc68ed0
//
void __fastcall sub_C68ED0(char *a1, char *a2)
{
  char *v2; // r15
  __int64 v3; // r14
  size_t v4; // r15
  size_t v5; // r13
  const void *v6; // rdi
  size_t v7; // rdx
  int v8; // eax
  char *i; // r15
  __int64 v10; // r12
  size_t v11; // rdx
  size_t v12; // rbx
  int v13; // eax
  char *v14; // r15
  void *s1; // [rsp+10h] [rbp-50h]
  char *v17; // [rsp+20h] [rbp-40h]
  char *v18; // [rsp+28h] [rbp-38h]

  if ( a1 != a2 && a2 != a1 + 8 )
  {
    v17 = a1 + 8;
    v18 = a1 + 16;
    while ( 1 )
    {
      v3 = *(_QWORD *)v17;
      v4 = *(_QWORD *)(*(_QWORD *)a1 + 16LL);
      v5 = *(_QWORD *)(*(_QWORD *)v17 + 16LL);
      v6 = *(const void **)(*(_QWORD *)v17 + 8LL);
      v7 = v4;
      if ( v5 <= v4 )
        v7 = *(_QWORD *)(*(_QWORD *)v17 + 16LL);
      if ( v7
        && (s1 = *(void **)(*(_QWORD *)v17 + 8LL),
            v8 = memcmp(v6, *(const void **)(*(_QWORD *)a1 + 8LL), v7),
            v6 = s1,
            v8) )
      {
        if ( v8 < 0 )
          goto LABEL_4;
LABEL_13:
        for ( i = v17; ; i -= 8 )
        {
          v10 = *((_QWORD *)i - 1);
          v11 = v5;
          v12 = *(_QWORD *)(v10 + 16);
          if ( v12 <= v5 )
            v11 = *(_QWORD *)(v10 + 16);
          if ( !v11 )
            break;
          v13 = memcmp(v6, *(const void **)(v10 + 8), v11);
          if ( !v13 )
            break;
          if ( v13 >= 0 )
            goto LABEL_20;
LABEL_23:
          *(_QWORD *)i = v10;
          v6 = *(const void **)(v3 + 8);
          v5 = *(_QWORD *)(v3 + 16);
        }
        if ( v12 != v5 && v12 > v5 )
          goto LABEL_23;
LABEL_20:
        *(_QWORD *)i = v3;
        v14 = v18;
        v17 += 8;
        v18 += 8;
        if ( a2 == v14 )
          return;
      }
      else
      {
        if ( v5 == v4 || v5 >= v4 )
          goto LABEL_13;
LABEL_4:
        v2 = v18;
        if ( a1 != v17 )
          memmove(a1 + 8, a1, v17 - a1);
        v17 += 8;
        v18 += 8;
        *(_QWORD *)a1 = v3;
        if ( a2 == v2 )
          return;
      }
    }
  }
}
