// Function: sub_1396CF0
// Address: 0x1396cf0
//
void **__fastcall sub_1396CF0(__int64 a1, __int64 a2, __int64 a3, void **a4)
{
  __int64 i; // r15
  __int64 v6; // rax
  size_t v7; // rdx
  size_t v8; // r13
  const void *v9; // rsi
  size_t v10; // rdx
  const void *v11; // rdi
  void *v12; // r10
  int v13; // eax
  __int64 v14; // rbx
  void ***v15; // r12
  void **v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // r13
  size_t v19; // rdx
  const void *v20; // rax
  size_t v21; // rdx
  size_t v22; // r9
  size_t v23; // r15
  int v24; // eax
  void **v25; // rax
  void *v26; // rdi
  void *v27; // r15
  __int64 v29; // rbx
  void **v30; // rax
  __int64 v32; // [rsp+10h] [rbp-60h]
  void *v33; // [rsp+20h] [rbp-50h]
  void *v34; // [rsp+20h] [rbp-50h]
  void *v35; // [rsp+20h] [rbp-50h]
  void *v36; // [rsp+20h] [rbp-50h]
  signed __int64 n; // [rsp+28h] [rbp-48h]
  size_t na; // [rsp+28h] [rbp-48h]

  n = (a3 - 1) / 2;
  v32 = a3 & 1;
  if ( a2 < n )
  {
    for ( i = a2; ; i = v14 )
    {
      v14 = 2 * (i + 1);
      v15 = (void ***)(a1 + 16 * (i + 1));
      v16 = *v15;
      v17 = (__int64)**(v15 - 1);
      if ( !v17 )
        goto LABEL_9;
      v34 = **v15;
      if ( !v34 )
        goto LABEL_8;
      v6 = sub_1649960(v17);
      v8 = v7;
      v9 = (const void *)v6;
      v11 = (const void *)sub_1649960(v34);
      v12 = (void *)v10;
      if ( v8 < v10 )
      {
        if ( v8 )
        {
          v36 = (void *)v10;
          v13 = memcmp(v11, v9, v8);
          v12 = v36;
          if ( v13 )
            goto LABEL_31;
LABEL_7:
          if ( v8 > (unsigned __int64)v12 )
            goto LABEL_8;
        }
      }
      else
      {
        if ( v10 )
        {
          v33 = (void *)v10;
          v13 = memcmp(v11, v9, v10);
          v12 = v33;
          if ( v13 )
          {
LABEL_31:
            if ( v13 < 0 )
            {
LABEL_8:
              --v14;
              v15 = (void ***)(a1 + 8 * v14);
              v16 = *v15;
              goto LABEL_9;
            }
            goto LABEL_32;
          }
        }
        if ( (void *)v8 != v12 )
          goto LABEL_7;
      }
LABEL_32:
      v16 = *v15;
LABEL_9:
      *(_QWORD *)(a1 + 8 * i) = v16;
      if ( v14 >= n )
      {
        if ( v32 )
        {
LABEL_15:
          v18 = (v14 - 1) / 2;
          if ( v14 <= a2 )
            goto LABEL_28;
          while ( 2 )
          {
            v15 = (void ***)(a1 + 8 * v18);
            v25 = *v15;
            v26 = *a4;
            v27 = **v15;
            if ( !*a4 )
            {
LABEL_27:
              v15 = (void ***)(a1 + 8 * v14);
              goto LABEL_28;
            }
            if ( v27 )
            {
              v35 = (void *)sub_1649960(v26);
              na = v19;
              v20 = (const void *)sub_1649960(v27);
              v22 = na;
              v23 = v21;
              if ( na < v21 )
              {
                if ( !na )
                  goto LABEL_27;
                v24 = memcmp(v20, v35, na);
                v22 = na;
                if ( v24 )
                  goto LABEL_41;
LABEL_22:
                if ( v22 <= v23 )
                  goto LABEL_27;
              }
              else
              {
                if ( !v21 || (v24 = memcmp(v20, v35, v21), v22 = na, !v24) )
                {
                  if ( v22 == v23 )
                    goto LABEL_27;
                  goto LABEL_22;
                }
LABEL_41:
                if ( v24 >= 0 )
                {
                  v15 = (void ***)(a1 + 8 * v14);
                  goto LABEL_28;
                }
              }
              v25 = *v15;
            }
            else if ( !v26 )
            {
              goto LABEL_27;
            }
            *(_QWORD *)(a1 + 8 * v14) = v25;
            v14 = v18;
            if ( a2 >= v18 )
              goto LABEL_28;
            v18 = (v18 - 1) / 2;
            continue;
          }
        }
LABEL_35:
        if ( (a3 - 2) / 2 == v14 )
        {
          v29 = 2 * v14 + 2;
          v30 = *(void ***)(a1 + 8 * v29 - 8);
          v14 = v29 - 1;
          *v15 = v30;
          v15 = (void ***)(a1 + 8 * v14);
        }
        goto LABEL_15;
      }
    }
  }
  v15 = (void ***)(a1 + 8 * a2);
  if ( (a3 & 1) == 0 )
  {
    v14 = a2;
    goto LABEL_35;
  }
LABEL_28:
  *v15 = a4;
  return a4;
}
