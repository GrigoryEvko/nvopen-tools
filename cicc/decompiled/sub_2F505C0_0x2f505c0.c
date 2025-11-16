// Function: sub_2F505C0
// Address: 0x2f505c0
//
__int64 __fastcall sub_2F505C0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  char *v4; // rdi
  __int64 v5; // r14
  int v7; // r13d
  unsigned __int64 v8; // rcx
  unsigned int v9; // edx
  __int64 v10; // r12
  unsigned int v12; // edx
  __int64 v13; // rdx
  __int64 *v14; // r15
  __int64 v15; // rax
  unsigned __int64 v16; // r10
  char *v17; // rax
  __int64 v18; // rax
  __int64 v19; // r11
  __int64 v20; // r9
  __int64 i; // r8
  char *v22; // rcx
  __int64 v23; // rsi
  char *v24; // rax
  char *v25; // rax
  int v26; // edx
  __int64 v27; // r12
  __int64 v28; // r8
  _QWORD *v29; // rax
  _QWORD *v30; // rsi
  char *v31; // rdx
  char *v32; // rcx
  __int64 v33; // [rsp+8h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 8);
  v4 = *(char **)a2;
  if ( *(_QWORD *)a2 == v3 )
    return 0;
  v5 = *(_QWORD *)(a1 + 32);
  v7 = ~*((_DWORD *)v4 + 1);
  v8 = *(unsigned int *)(v5 + 160);
  v9 = v7 & 0x7FFFFFFF;
  if ( (v7 & 0x7FFFFFFFu) >= (unsigned int)v8 || (v10 = *(_QWORD *)(*(_QWORD *)(v5 + 152) + 8LL * v9)) == 0 )
  {
    v12 = v9 + 1;
    if ( (unsigned int)v8 < v12 && v12 != v8 )
    {
      if ( v12 >= v8 )
      {
        v27 = *(_QWORD *)(v5 + 168);
        v28 = v12 - v8;
        if ( v12 > (unsigned __int64)*(unsigned int *)(v5 + 164) )
        {
          v33 = v12 - v8;
          sub_C8D5F0(v5 + 152, (const void *)(v5 + 168), v12, 8u, v28, v12);
          v8 = *(unsigned int *)(v5 + 160);
          v28 = v33;
        }
        v13 = *(_QWORD *)(v5 + 152);
        v29 = (_QWORD *)(v13 + 8 * v8);
        v30 = &v29[v28];
        if ( v29 != v30 )
        {
          do
            *v29++ = v27;
          while ( v30 != v29 );
          LODWORD(v8) = *(_DWORD *)(v5 + 160);
          v13 = *(_QWORD *)(v5 + 152);
        }
        *(_DWORD *)(v5 + 160) = v28 + v8;
LABEL_9:
        v14 = (__int64 *)(v13 + 8LL * (v7 & 0x7FFFFFFF));
        v15 = sub_2E10F30(v7);
        *v14 = v15;
        v10 = v15;
        sub_2E11E80((_QWORD *)v5, v15);
        v3 = *(_QWORD *)(a2 + 8);
        v4 = *(char **)a2;
        if ( v3 - *(_QWORD *)a2 <= 8 )
          goto LABEL_5;
LABEL_10:
        v16 = *(_QWORD *)(v3 - 8);
        v17 = (char *)(v3 - 8);
        *(_DWORD *)v17 = *(_DWORD *)v4;
        *((_DWORD *)v17 + 1) = *((_DWORD *)v4 + 1);
        v18 = v17 - v4;
        v19 = v18 >> 3;
        v20 = ((v18 >> 3) - 1) / 2;
        if ( v18 <= 16 )
        {
          v23 = 0;
        }
        else
        {
          for ( i = 0; ; i = v23 )
          {
            v23 = 2 * (i + 1) - 1;
            v25 = &v4[16 * i + 16];
            v22 = &v4[8 * v23];
            v26 = *(_DWORD *)v22;
            if ( *(_DWORD *)v25 >= *(_DWORD *)v22 )
            {
              if ( *(_DWORD *)v25 == v26 )
              {
                if ( *((_DWORD *)v25 + 1) >= *((_DWORD *)v22 + 1) )
                {
                  v22 = &v4[16 * i + 16];
                  v23 = 2 * (i + 1);
                }
              }
              else
              {
                v26 = *(_DWORD *)v25;
                v22 = &v4[16 * i + 16];
                v23 = 2 * (i + 1);
              }
            }
            v24 = &v4[8 * i];
            *(_DWORD *)v24 = v26;
            *((_DWORD *)v24 + 1) = *((_DWORD *)v22 + 1);
            if ( v23 >= v20 )
              break;
          }
        }
        if ( (v19 & 1) == 0 && v23 == (v19 - 2) / 2 )
        {
          v31 = &v4[8 * v23];
          v32 = &v4[16 * v23 + 8];
          *(_DWORD *)v31 = *(_DWORD *)v32;
          v23 = 2 * v23 + 1;
          *((_DWORD *)v31 + 1) = *((_DWORD *)v32 + 1);
        }
        sub_2F4CA50((__int64)v4, v23, 0, v16);
        v3 = *(_QWORD *)(a2 + 8);
        goto LABEL_5;
      }
      *(_DWORD *)(v5 + 160) = v12;
    }
    v13 = *(_QWORD *)(v5 + 152);
    goto LABEL_9;
  }
  if ( v3 - (__int64)v4 > 8 )
    goto LABEL_10;
LABEL_5:
  *(_QWORD *)(a2 + 8) = v3 - 8;
  return v10;
}
