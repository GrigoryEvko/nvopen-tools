// Function: sub_1A915D0
// Address: 0x1a915d0
//
void __fastcall sub_1A915D0(_QWORD *a1, _QWORD *a2)
{
  int v2; // eax
  __int64 v3; // r14
  _QWORD *v4; // r12
  __int64 v5; // r12
  const char *v6; // r15
  size_t v7; // rdx
  size_t v8; // r14
  size_t v9; // rdx
  const char *v10; // rdi
  size_t v11; // r12
  _QWORD *v12; // r14
  __int64 v13; // r12
  int v14; // eax
  __int64 v15; // rax
  const char *v16; // r15
  size_t v17; // rdx
  size_t v18; // r13
  size_t v19; // rdx
  const char *v20; // rdi
  size_t v21; // rbx
  _QWORD *v23; // [rsp+18h] [rbp-38h]

  if ( a1 != a2 && a1 + 1 != a2 )
  {
    v23 = a1 + 1;
    do
    {
      while ( 1 )
      {
        v5 = *v23;
        v6 = sub_1649960(*(_QWORD *)(*a1 + 40LL));
        v8 = v7;
        v10 = sub_1649960(*(_QWORD *)(v5 + 40));
        v11 = v9;
        if ( v9 <= v8 )
          break;
        if ( !v8 )
          goto LABEL_15;
        v2 = memcmp(v10, v6, v8);
        if ( v2 )
          goto LABEL_14;
LABEL_7:
        if ( v11 < v8 )
          goto LABEL_8;
LABEL_15:
        v12 = v23;
        v13 = *v23;
        while ( 1 )
        {
          v16 = sub_1649960(*(_QWORD *)(*(v12 - 1) + 40LL));
          v18 = v17;
          v20 = sub_1649960(*(_QWORD *)(v13 + 40));
          v21 = v19;
          if ( v19 > v18 )
            break;
          if ( v19 )
          {
            v14 = memcmp(v20, v16, v19);
            if ( v14 )
              goto LABEL_24;
          }
          if ( v21 == v18 )
            goto LABEL_25;
LABEL_19:
          if ( v21 >= v18 )
            goto LABEL_25;
LABEL_20:
          v15 = *--v12;
          v12[1] = v15;
        }
        if ( !v18 )
          goto LABEL_25;
        v14 = memcmp(v20, v16, v18);
        if ( !v14 )
          goto LABEL_19;
LABEL_24:
        if ( v14 < 0 )
          goto LABEL_20;
LABEL_25:
        *v12 = v13;
        if ( a2 == ++v23 )
          return;
      }
      if ( !v9 || (v2 = memcmp(v10, v6, v9)) == 0 )
      {
        if ( v11 == v8 )
          goto LABEL_15;
        goto LABEL_7;
      }
LABEL_14:
      if ( v2 >= 0 )
        goto LABEL_15;
LABEL_8:
      v3 = *v23;
      v4 = v23 + 1;
      if ( a1 != v23 )
        memmove(a1 + 1, a1, (char *)v23 - (char *)a1);
      ++v23;
      *a1 = v3;
    }
    while ( a2 != v4 );
  }
}
