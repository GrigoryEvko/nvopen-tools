// Function: sub_1D8D810
// Address: 0x1d8d810
//
void __fastcall sub_1D8D810(unsigned __int64 *a1, __int64 a2)
{
  unsigned __int64 v2; // rcx
  __int64 *v3; // rbx
  unsigned __int64 *i; // rdx
  unsigned __int64 *v5; // rax
  unsigned __int64 *v7; // rax
  __int64 *v8; // r12
  unsigned __int64 *v9; // rcx
  unsigned int v11; // r9d
  unsigned int v12; // edx
  __int64 v13; // rax
  __int64 *v14; // rsi
  unsigned int v15; // esi
  __int64 v16; // rdx
  unsigned __int64 *v17; // r11
  unsigned int v18; // edi
  __int64 v19; // r8
  unsigned __int64 *v20; // rax
  unsigned int v21; // ebx
  unsigned __int64 **v22; // r9
  unsigned int v23; // esi
  __int64 v24; // rdx
  char **v25; // r11
  unsigned __int64 *v26; // r11
  __int64 v27; // rsi
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rsi
  unsigned __int64 v31; // rcx
  unsigned __int64 v32; // rdx
  __int64 v33; // rcx
  int v34; // esi
  int v35; // r11d
  __int64 v36; // rsi
  int v37; // edx
  int v38; // edx
  int v39; // eax
  int v40; // r10d
  int v41; // r10d
  int v42; // [rsp-5Ch] [rbp-5Ch]
  unsigned int v43; // [rsp-58h] [rbp-58h]
  int v44; // [rsp-58h] [rbp-58h]
  unsigned int v45; // [rsp-54h] [rbp-54h]
  unsigned __int64 *v46; // [rsp-50h] [rbp-50h]
  unsigned __int64 v47; // [rsp-48h] [rbp-48h] BYREF
  __int64 *v48; // [rsp-40h] [rbp-40h]

  v2 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1 != (unsigned __int64 *)v2 )
  {
    v3 = (__int64 *)a1;
    i = (unsigned __int64 *)a1[1];
    v5 = (unsigned __int64 *)i[1];
    if ( a1 != v5 )
    {
      if ( a1 != i )
      {
        for ( i = (unsigned __int64 *)i[1]; ; i = (unsigned __int64 *)i[1] )
        {
          v7 = (unsigned __int64 *)v5[1];
          if ( a1 == v7 )
            break;
          v5 = (unsigned __int64 *)v7[1];
          if ( a1 == v5 )
            break;
        }
      }
      v48 = (__int64 *)&v47;
      v47 = (unsigned __int64)&v47 + 4;
      if ( a1 != i )
      {
        *(_QWORD *)((*i & 0xFFFFFFFFFFFFFFF8LL) + 8) = a1;
        *a1 = *a1 & 7 | *i & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v2 + 8) = &v47;
        *i = (unsigned __int64)&v47 | *i & 7;
        v48 = (__int64 *)i;
        v47 = v47 & 7 | v2;
      }
      sub_1D8D810(a1, a2);
      sub_1D8D810(&v47, a2);
      if ( &v47 != (unsigned __int64 *)(v47 & 0xFFFFFFFFFFFFFFF8LL) )
      {
        v8 = v48;
        v9 = (unsigned __int64 *)a1[1];
        v46 = (unsigned __int64 *)v48;
        if ( a1 != v9 )
        {
          do
          {
            while ( 1 )
            {
              v18 = *(_DWORD *)(a2 + 24);
              v19 = *(_QWORD *)(a2 + 8);
              if ( v18 )
              {
                v11 = v18 - 1;
                v12 = (v18 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
                v13 = v19 + 16LL * v12;
                v14 = *(__int64 **)v13;
                if ( *(__int64 **)v13 != v8 )
                {
                  v39 = 1;
                  while ( v14 != (__int64 *)-8LL )
                  {
                    v41 = v39 + 1;
                    v12 = v11 & (v39 + v12);
                    v13 = v19 + 16LL * v12;
                    v14 = *(__int64 **)v13;
                    if ( *(__int64 **)v13 == v8 )
                      goto LABEL_14;
                    v39 = v41;
                  }
                  v13 = v19 + 16LL * v18;
                }
LABEL_14:
                v15 = v11 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
                v16 = v19 + 16LL * v15;
                v17 = *(unsigned __int64 **)v16;
                if ( v9 != *(unsigned __int64 **)v16 )
                {
                  v38 = 1;
                  while ( v17 != (unsigned __int64 *)-8LL )
                  {
                    v40 = v38 + 1;
                    v15 = v11 & (v38 + v15);
                    v16 = v19 + 16LL * v15;
                    v17 = *(unsigned __int64 **)v16;
                    if ( *(unsigned __int64 **)v16 == v9 )
                      goto LABEL_15;
                    v38 = v40;
                  }
                  v16 = v19 + 16LL * v18;
                }
LABEL_15:
                if ( *(_DWORD *)(v13 + 8) < *(_DWORD *)(v16 + 8) )
                  break;
              }
              v9 = (unsigned __int64 *)v9[1];
              if ( a1 == v9 )
                goto LABEL_32;
            }
            v20 = (unsigned __int64 *)v8[1];
            if ( v20 != &v47 )
            {
              v21 = v18 - 1;
              v45 = (v18 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
              v22 = (unsigned __int64 **)(v19 + 16LL * v45);
              do
              {
                v23 = v21 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
                v24 = v19 + 16LL * v23;
                v25 = *(char ***)v24;
                if ( *(unsigned __int64 **)v24 != v20 )
                {
                  v37 = 1;
                  while ( v25 != (char **)-8LL )
                  {
                    v23 = v21 & (v37 + v23);
                    v44 = v37 + 1;
                    v24 = v19 + 16LL * v23;
                    v25 = *(char ***)v24;
                    if ( *(unsigned __int64 **)v24 == v20 )
                      goto LABEL_22;
                    v37 = v44;
                  }
                  v24 = v19 + 16LL * v18;
                }
LABEL_22:
                v26 = *v22;
                v27 = v19 + 16LL * v45;
                if ( v9 == *v22 )
                {
LABEL_23:
                  if ( *(_DWORD *)(v24 + 8) >= *(_DWORD *)(v27 + 8) )
                    goto LABEL_27;
                }
                else
                {
                  v43 = (v18 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
                  v34 = 1;
                  while ( v26 != (unsigned __int64 *)-8LL )
                  {
                    v35 = v34 + 1;
                    v36 = v21 & (v43 + v34);
                    v42 = v35;
                    v43 = v36;
                    v27 = v19 + 16 * v36;
                    v26 = *(unsigned __int64 **)v27;
                    if ( v9 == *(unsigned __int64 **)v27 )
                      goto LABEL_23;
                    v34 = v42;
                  }
                  if ( *(_DWORD *)(v24 + 8) >= *(_DWORD *)(v19 + 16LL * v18 + 8) )
                    goto LABEL_27;
                }
                v20 = (unsigned __int64 *)v20[1];
              }
              while ( v20 != &v47 );
            }
            v20 = &v47;
LABEL_27:
            if ( v20 != v9 && v20 != v46 )
            {
              v28 = *v20 & 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)((*v8 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v20;
              *v20 = *v20 & 7 | *v8 & 0xFFFFFFFFFFFFFFF8LL;
              v29 = *v9;
              *(_QWORD *)(v28 + 8) = v9;
              v29 &= 0xFFFFFFFFFFFFFFF8LL;
              *v8 = v29 | *v8 & 7;
              *(_QWORD *)(v29 + 8) = v46;
              *v9 = v28 | *v9 & 7;
            }
            if ( v20 == &v47 )
              return;
            v9 = (unsigned __int64 *)v9[1];
            v46 = v20;
            v8 = (__int64 *)v20;
          }
          while ( a1 != v9 );
LABEL_32:
          v3 = (__int64 *)a1;
        }
        if ( v46 != &v47 )
        {
          v30 = v47;
          v31 = v47 & 7;
          *(_QWORD *)((*v8 & 0xFFFFFFFFFFFFFFF8LL) + 8) = &v47;
          v30 &= 0xFFFFFFFFFFFFFFF8LL;
          v32 = v31 | *v8 & 0xFFFFFFFFFFFFFFF8LL;
          v33 = *v3;
          *(_QWORD *)(v30 + 8) = v3;
          v47 = v32;
          v33 &= 0xFFFFFFFFFFFFFFF8LL;
          *v8 = v33 | *v8 & 7;
          *(_QWORD *)(v33 + 8) = v46;
          *v3 = v30 | *v3 & 7;
        }
      }
    }
  }
}
