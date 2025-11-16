// Function: sub_34E54F0
// Address: 0x34e54f0
//
void __fastcall sub_34E54F0(unsigned __int64 *a1, __int64 a2)
{
  unsigned __int64 v2; // rcx
  __int64 *v3; // rbx
  unsigned __int64 *i; // rdx
  unsigned __int64 *v5; // rax
  unsigned __int64 *v7; // rax
  __int64 *v8; // r11
  unsigned __int64 *v9; // rcx
  unsigned int v11; // esi
  __int64 v12; // r9
  unsigned int v13; // r8d
  unsigned int v14; // edx
  __int64 v15; // rax
  __int64 *v16; // rdi
  unsigned int v17; // edi
  __int64 v18; // rdx
  unsigned __int64 *v19; // rbx
  unsigned __int64 *v20; // rax
  unsigned int v21; // ebx
  unsigned __int64 **v22; // rdi
  unsigned int v23; // r8d
  __int64 v24; // rdx
  char **v25; // r10
  unsigned __int64 *v26; // r10
  __int64 v27; // r8
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rsi
  unsigned __int64 v31; // rcx
  unsigned __int64 v32; // rdx
  __int64 v33; // rcx
  int v34; // r8d
  int v35; // r10d
  __int64 v36; // r8
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
  if ( (unsigned __int64 *)v2 != a1 )
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
      sub_34E54F0(a1, a2);
      sub_34E54F0(&v47, a2);
      if ( &v47 != (unsigned __int64 *)(v47 & 0xFFFFFFFFFFFFFFF8LL) )
      {
        v8 = v48;
        v9 = (unsigned __int64 *)a1[1];
        v46 = (unsigned __int64 *)v48;
        if ( a1 != v9 )
        {
          while ( 1 )
          {
            while ( 1 )
            {
              v11 = *(_DWORD *)(a2 + 24);
              v12 = *(_QWORD *)(a2 + 8);
              if ( v11 )
              {
                v13 = v11 - 1;
                v14 = (v11 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
                v15 = v12 + 16LL * v14;
                v16 = *(__int64 **)v15;
                if ( v8 != *(__int64 **)v15 )
                {
                  v39 = 1;
                  while ( v16 != (__int64 *)-4096LL )
                  {
                    v40 = v39 + 1;
                    v14 = v13 & (v39 + v14);
                    v15 = v12 + 16LL * v14;
                    v16 = *(__int64 **)v15;
                    if ( v8 == *(__int64 **)v15 )
                      goto LABEL_16;
                    v39 = v40;
                  }
                  v15 = v12 + 16LL * v11;
                }
LABEL_16:
                v17 = v13 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
                v18 = v12 + 16LL * v17;
                v19 = *(unsigned __int64 **)v18;
                if ( v9 != *(unsigned __int64 **)v18 )
                {
                  v38 = 1;
                  while ( v19 != (unsigned __int64 *)-4096LL )
                  {
                    v41 = v38 + 1;
                    v17 = v13 & (v38 + v17);
                    v18 = v12 + 16LL * v17;
                    v19 = *(unsigned __int64 **)v18;
                    if ( v9 == *(unsigned __int64 **)v18 )
                      goto LABEL_17;
                    v38 = v41;
                  }
                  v18 = v12 + 16LL * v11;
                }
              }
              else
              {
                v18 = *(_QWORD *)(a2 + 8);
                v15 = v18;
              }
LABEL_17:
              if ( *(_DWORD *)(v15 + 8) < *(_DWORD *)(v18 + 8) )
                break;
              v9 = (unsigned __int64 *)v9[1];
              if ( a1 == v9 )
                goto LABEL_32;
            }
            v20 = (unsigned __int64 *)v8[1];
            if ( v20 != &v47 )
              break;
LABEL_44:
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
            if ( a1 == v9 )
            {
LABEL_32:
              v3 = (__int64 *)a1;
              goto LABEL_33;
            }
          }
          v21 = v11 - 1;
          v45 = (v11 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
          v22 = (unsigned __int64 **)(v12 + 16LL * v45);
          while ( 1 )
          {
            if ( v11 )
            {
              v23 = v21 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
              v24 = v12 + 16LL * v23;
              v25 = *(char ***)v24;
              if ( v20 != *(unsigned __int64 **)v24 )
              {
                v37 = 1;
                while ( v25 != (char **)-4096LL )
                {
                  v23 = v21 & (v37 + v23);
                  v44 = v37 + 1;
                  v24 = v12 + 16LL * v23;
                  v25 = *(char ***)v24;
                  if ( v20 == *(unsigned __int64 **)v24 )
                    goto LABEL_21;
                  v37 = v44;
                }
                v24 = v12 + 16LL * v11;
              }
LABEL_21:
              v26 = *v22;
              v27 = v12 + 16LL * v45;
              if ( v9 == *v22 )
              {
LABEL_22:
                if ( *(_DWORD *)(v24 + 8) >= *(_DWORD *)(v27 + 8) )
                  goto LABEL_27;
                goto LABEL_23;
              }
              v43 = (v11 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
              v34 = 1;
              while ( v26 != (unsigned __int64 *)-4096LL )
              {
                v35 = v34 + 1;
                v36 = v21 & (v43 + v34);
                v42 = v35;
                v43 = v36;
                v27 = v12 + 16 * v36;
                v26 = *(unsigned __int64 **)v27;
                if ( v9 == *(unsigned __int64 **)v27 )
                  goto LABEL_22;
                v34 = v42;
              }
            }
            else
            {
              v24 = *(_QWORD *)(a2 + 8);
            }
            if ( *(_DWORD *)(v24 + 8) >= *(_DWORD *)(v12 + 16LL * v11 + 8) )
              goto LABEL_27;
LABEL_23:
            v20 = (unsigned __int64 *)v20[1];
            if ( v20 == &v47 )
              goto LABEL_44;
          }
        }
LABEL_33:
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
