// Function: sub_23B7BC0
// Address: 0x23b7bc0
//
void __fastcall sub_23B7BC0(__int64 **a1, __int64 *a2, __int64 a3)
{
  __int64 **v3; // r15
  unsigned __int64 v4; // r12
  __int64 *v5; // rbx
  __int64 v6; // r13
  __int64 i; // r12
  __int64 *v8; // rcx
  __int64 v9; // rdi
  unsigned int v10; // eax
  __int64 v11; // r13
  __int64 *v12; // r14
  __int64 v13; // rax
  __int64 v14; // r15
  size_t v15; // r12
  _QWORD *v16; // rbx
  __int64 v17; // r12
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // r15
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // rax
  size_t v24; // rdx
  _QWORD *v25; // r14
  __int64 v26; // rbx
  _QWORD *v27; // r12
  __int64 v28; // rcx
  __int64 v29; // [rsp+0h] [rbp-A0h]
  __int64 *v30; // [rsp+10h] [rbp-90h]
  __int64 v31; // [rsp+18h] [rbp-88h]
  __int64 v32; // [rsp+20h] [rbp-80h]
  __int64 **v33; // [rsp+28h] [rbp-78h]
  __int64 v34; // [rsp+30h] [rbp-70h]
  __int64 v35; // [rsp+38h] [rbp-68h]
  _QWORD *j; // [rsp+40h] [rbp-60h]
  __int64 v37; // [rsp+48h] [rbp-58h]
  size_t n; // [rsp+50h] [rbp-50h]
  __int64 v39; // [rsp+58h] [rbp-48h]
  _QWORD *v40; // [rsp+68h] [rbp-38h]

  v3 = a1;
  v4 = a2[1] - *a2;
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( v4 )
  {
    if ( v4 > 0x7FFFFFFFFFFFFFE0LL )
      sub_4261EA(a1, a2, a3);
    v5 = (__int64 *)sub_22077B0(v4);
  }
  else
  {
    v4 = 0;
    v5 = 0;
  }
  *a1 = v5;
  a1[1] = v5;
  a1[2] = (__int64 *)((char *)v5 + v4);
  v6 = a2[1];
  for ( i = *a2; v6 != i; v5 += 4 )
  {
    if ( v5 )
    {
      *v5 = (__int64)(v5 + 2);
      sub_23AEDD0(v5, *(_BYTE **)i, *(_QWORD *)i + *(_QWORD *)(i + 8));
    }
    i += 32;
  }
  a1[1] = v5;
  a1[3] = 0;
  a1[4] = 0;
  a1[5] = (__int64 *)0x6000000000LL;
  if ( *((_DWORD *)a2 + 9) )
  {
    sub_C92620((__int64)(a1 + 3), *((_DWORD *)a2 + 8));
    v8 = a1[3];
    v9 = a2[3];
    v10 = *((_DWORD *)v3 + 8);
    v30 = v3[3];
    v39 = 8LL * v10 + 8;
    *(__int64 **)((char *)v3 + 36) = *(__int64 **)((char *)a2 + 36);
    if ( v10 )
    {
      v33 = v3;
      v11 = 0;
      v12 = v8;
      v34 = 8LL * (v10 - 1);
      v13 = v9;
      while ( 1 )
      {
        v14 = *(_QWORD *)(v13 + v11);
        v40 = (__int64 *)((char *)v12 + v11);
        if ( !v14 || v14 == -8 )
        {
          v39 += 4;
          *v40 = v14;
          if ( v34 == v11 )
            goto LABEL_30;
        }
        else
        {
          v15 = *(_QWORD *)v14;
          v16 = (_QWORD *)sub_C7D670(*(_QWORD *)v14 + 97LL, 8);
          if ( v15 )
            memcpy(v16 + 12, (const void *)(v14 + 96), v15);
          *((_BYTE *)v16 + v15 + 96) = 0;
          v16[1] = v16 + 3;
          *v16 = v15;
          sub_23AEDD0(v16 + 1, *(_BYTE **)(v14 + 8), *(_QWORD *)(v14 + 8) + *(_QWORD *)(v14 + 16));
          v16[5] = v16 + 7;
          sub_23AEDD0(v16 + 5, *(_BYTE **)(v14 + 40), *(_QWORD *)(v14 + 40) + *(_QWORD *)(v14 + 48));
          v16[9] = 0;
          v16[10] = 0;
          v16[11] = 0x2800000000LL;
          if ( *(_DWORD *)(v14 + 84) )
          {
            sub_C92620((__int64)(v16 + 9), *(_DWORD *)(v14 + 80));
            v17 = v16[9];
            v18 = *(_QWORD *)(v14 + 72);
            v19 = *((unsigned int *)v16 + 20);
            v32 = v17;
            v31 = v18;
            *(_QWORD *)((char *)v16 + 84) = *(_QWORD *)(v14 + 84);
            if ( (_DWORD)v19 )
            {
              v35 = v14;
              v20 = 8 * v19 + 8;
              v29 = v11;
              v21 = 0;
              v37 = 8LL * (unsigned int)(v19 - 1);
              v22 = v18;
              for ( j = v16; ; v17 = j[9] )
              {
                v26 = *(_QWORD *)(v22 + v21);
                v27 = (_QWORD *)(v21 + v17);
                if ( v26 == -8 || !v26 )
                {
                  *v27 = v26;
                  v20 += 4;
                  if ( v21 == v37 )
                    goto LABEL_26;
                }
                else
                {
                  n = *(_QWORD *)v26;
                  v23 = sub_C7D670(*(_QWORD *)v26 + 41LL, 8);
                  v24 = n;
                  v25 = (_QWORD *)v23;
                  if ( n )
                  {
                    memcpy((void *)(v23 + 40), (const void *)(v26 + 40), n);
                    v24 = n;
                  }
                  *((_BYTE *)v25 + v24 + 40) = 0;
                  v25[1] = v25 + 3;
                  *v25 = v24;
                  sub_23AEDD0(v25 + 1, *(_BYTE **)(v26 + 8), *(_QWORD *)(v26 + 8) + *(_QWORD *)(v26 + 16));
                  *v27 = v25;
                  *(_DWORD *)(v32 + v20) = *(_DWORD *)(v31 + v20);
                  v20 += 4;
                  if ( v21 == v37 )
                  {
LABEL_26:
                    v16 = j;
                    v11 = v29;
                    break;
                  }
                }
                v21 += 8;
                v22 = *(_QWORD *)(v35 + 72);
              }
            }
          }
          v28 = v39;
          v39 += 4;
          *v40 = v16;
          *(_DWORD *)((char *)v30 + v28) = *(_DWORD *)(v9 + v28);
          if ( v34 == v11 )
          {
LABEL_30:
            v3 = v33;
            break;
          }
        }
        v11 += 8;
        v13 = a2[3];
        v12 = v33[3];
      }
    }
  }
  v3[6] = (__int64 *)(v3 + 8);
  sub_23AEDD0((__int64 *)v3 + 6, (_BYTE *)a2[6], a2[6] + a2[7]);
}
