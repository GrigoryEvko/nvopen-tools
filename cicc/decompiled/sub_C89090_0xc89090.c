// Function: sub_C89090
// Address: 0xc89090
//
__int64 __fastcall sub_C89090(_QWORD *a1, const char *a2, __int64 a3, __int64 a4, _QWORD *a5)
{
  const char *v5; // r10
  __int64 v9; // rdx
  char v10; // al
  unsigned int v11; // eax
  unsigned int v12; // r9d
  __int64 v13; // r11
  unsigned __int64 v14; // rdx
  __int64 v15; // r8
  char *v16; // rdx
  _QWORD *v17; // rcx
  char *v18; // r8
  __int64 v19; // rsi
  _BYTE *v20; // rcx
  unsigned int v21; // r14d
  int v22; // eax
  const char *v23; // r10
  __int64 v24; // r13
  __int64 v25; // rax
  __int64 v26; // r8
  const char *v27; // r14
  const char *v28; // r12
  const char **v29; // rax
  unsigned __int64 v30; // rcx
  unsigned __int64 v31; // r9
  __int64 v32; // r12
  unsigned int v34; // ebx
  __int64 v35; // r13
  const char *v36; // [rsp+8h] [rbp-E8h]
  __int64 v37; // [rsp+10h] [rbp-E0h]
  __int64 v38; // [rsp+18h] [rbp-D8h]
  __int64 v39; // [rsp+18h] [rbp-D8h]
  __int64 v40; // [rsp+18h] [rbp-D8h]
  unsigned int v41; // [rsp+20h] [rbp-D0h]
  const char *v42; // [rsp+20h] [rbp-D0h]
  unsigned int v43; // [rsp+20h] [rbp-D0h]
  const char *v44; // [rsp+20h] [rbp-D0h]
  const char *v45; // [rsp+28h] [rbp-C8h]
  const char *v46; // [rsp+28h] [rbp-C8h]
  __int64 v47; // [rsp+28h] [rbp-C8h]
  unsigned int v48; // [rsp+28h] [rbp-C8h]
  _BYTE *v49; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v50; // [rsp+38h] [rbp-B8h]
  _BYTE v51[16]; // [rsp+40h] [rbp-B0h] BYREF
  char v52; // [rsp+50h] [rbp-A0h] BYREF

  v5 = a2;
  if ( !a5 )
  {
    if ( !*((_DWORD *)a1 + 2) )
      goto LABEL_5;
    return 0;
  }
  v9 = a5[1];
  if ( v9 )
  {
    sub_2241130(a5, 0, v9, byte_3F871B3, 0);
    v5 = a2;
  }
  v45 = v5;
  v10 = sub_C89030(a1, a5);
  v5 = v45;
  if ( !v10 )
    return 0;
LABEL_5:
  if ( a4 )
  {
    v11 = 1;
    v12 = *(_QWORD *)(*a1 + 8LL) + 1;
    if ( (unsigned int)*(_QWORD *)(*a1 + 8LL) != -1 )
      v11 = *(_QWORD *)(*a1 + 8LL) + 1;
    v13 = v12;
    v14 = v11;
    v15 = 16LL * v11;
    if ( !a2 )
    {
      a3 = 0;
      v5 = byte_3F871B3;
    }
  }
  else
  {
    if ( !a2 )
    {
      v13 = 0;
      a3 = 0;
      v12 = 0;
      v49 = v51;
      v16 = v51;
      v18 = &v52;
      v50 = 0x800000000LL;
      v5 = byte_3F871B3;
      v11 = 1;
      do
      {
LABEL_13:
        if ( v16 )
        {
          *(_QWORD *)v16 = 0;
          *((_QWORD *)v16 + 1) = 0;
        }
        v16 += 16;
      }
      while ( v16 != v18 );
      v17 = v49;
      goto LABEL_17;
    }
    v13 = 0;
    v14 = 1;
    v11 = 1;
    v12 = 0;
    v15 = 16;
  }
  v49 = v51;
  v50 = 0x800000000LL;
  if ( v14 > 8 )
  {
    v36 = v5;
    v37 = v15;
    v39 = v13;
    v43 = v12;
    v48 = v11;
    sub_C8D5F0(&v49, v51, v14, 16);
    v17 = v49;
    v5 = v36;
    v15 = v37;
    v13 = v39;
    v12 = v43;
    v11 = v48;
    v16 = &v49[16 * (unsigned int)v50];
  }
  else
  {
    v16 = v51;
    v17 = v51;
  }
  v18 = (char *)v17 + v15;
  if ( v18 != v16 )
    goto LABEL_13;
LABEL_17:
  LODWORD(v50) = v11;
  v19 = (__int64)v5;
  *v17 = 0;
  v20 = v49;
  v41 = v12;
  *((_QWORD *)v49 + 1) = a3;
  v21 = 0;
  v46 = v5;
  v22 = sub_CBDA10(*a1, v5, v13, v20, 4);
  if ( v22 != 1 )
  {
    v23 = v46;
    if ( v22 )
    {
      if ( a5 )
      {
        v34 = *((_DWORD *)a1 + 2);
        v47 = *a1;
        v35 = sub_CBADF0(v34, *a1, 0, 0);
        sub_22410F0(a5, v35 - 1, 0);
        v19 = v47;
        sub_CBADF0(v34, v47, *a5, v35);
      }
    }
    else
    {
      if ( a4 )
      {
        *(_DWORD *)(a4 + 8) = 0;
        if ( v41 )
        {
          v24 = 0;
          v19 = a4 + 16;
          v25 = 0;
          v26 = 16LL * (v41 - 1);
          while ( 1 )
          {
            v30 = *(unsigned int *)(a4 + 12);
            v31 = v25 + 1;
            v32 = *(_QWORD *)&v49[v24];
            if ( v32 == -1 )
            {
              if ( v31 > v30 )
              {
                v40 = v26;
                v44 = v23;
                sub_C8D5F0(a4, v19, v25 + 1, 16);
                v25 = *(unsigned int *)(a4 + 8);
                v26 = v40;
                v23 = v44;
              }
              *(_OWORD *)(*(_QWORD *)a4 + 16 * v25) = 0;
              ++*(_DWORD *)(a4 + 8);
              if ( v24 == v26 )
                break;
            }
            else
            {
              v27 = (const char *)(*(_QWORD *)&v49[v24 + 8] - v32);
              v28 = &v23[v32];
              if ( v31 > v30 )
              {
                v38 = v26;
                v42 = v23;
                sub_C8D5F0(a4, v19, v25 + 1, 16);
                v25 = *(unsigned int *)(a4 + 8);
                v26 = v38;
                v23 = v42;
              }
              v29 = (const char **)(*(_QWORD *)a4 + 16 * v25);
              *v29 = v28;
              v29[1] = v27;
              ++*(_DWORD *)(a4 + 8);
              if ( v24 == v26 )
                break;
            }
            v25 = *(unsigned int *)(a4 + 8);
            v24 += 16;
          }
        }
      }
      v21 = 1;
    }
  }
  if ( v49 != v51 )
    _libc_free(v49, v19);
  return v21;
}
