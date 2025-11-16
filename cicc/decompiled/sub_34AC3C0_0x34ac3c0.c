// Function: sub_34AC3C0
// Address: 0x34ac3c0
//
void __fastcall sub_34AC3C0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 a6,
        int a7,
        __int64 a8,
        unsigned int a9)
{
  __int64 v10; // r15
  const __m128i *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  char *v16; // rax
  char *v17; // rcx
  int v18; // edx
  unsigned __int64 v19; // rdi
  __int64 v20; // r14
  __int64 v21; // r13
  char *v22; // rax
  char *v23; // rcx
  int v24; // edx
  char *v25; // rax
  char *v26; // rcx
  int v27; // edx
  int v30; // [rsp+18h] [rbp-4F8h]
  int v31; // [rsp+20h] [rbp-4F0h] BYREF
  __int64 v32; // [rsp+28h] [rbp-4E8h]
  __int64 v33; // [rsp+30h] [rbp-4E0h]
  __int64 v34; // [rsp+40h] [rbp-4D0h] BYREF
  __int64 v35; // [rsp+48h] [rbp-4C8h]
  __int64 v36; // [rsp+50h] [rbp-4C0h]
  __int64 v37; // [rsp+58h] [rbp-4B8h]
  __m128i v38[4]; // [rsp+60h] [rbp-4B0h] BYREF
  char *v39; // [rsp+A0h] [rbp-470h]
  int v40; // [rsp+A8h] [rbp-468h]
  char v41; // [rsp+B0h] [rbp-460h] BYREF
  char *v42; // [rsp+1B0h] [rbp-360h]
  char v43; // [rsp+1C0h] [rbp-350h] BYREF
  __m128i v44[4]; // [rsp+1E0h] [rbp-330h] BYREF
  char *v45; // [rsp+220h] [rbp-2F0h]
  int v46; // [rsp+228h] [rbp-2E8h]
  char v47; // [rsp+230h] [rbp-2E0h] BYREF
  char *v48; // [rsp+330h] [rbp-1E0h]
  char v49; // [rsp+340h] [rbp-1D0h] BYREF
  __m128i v50[4]; // [rsp+360h] [rbp-1B0h] BYREF
  char *v51; // [rsp+3A0h] [rbp-170h]
  int v52; // [rsp+3A8h] [rbp-168h]
  char v53; // [rsp+3B0h] [rbp-160h] BYREF
  char *v54; // [rsp+4B0h] [rbp-60h]
  char v55; // [rsp+4C0h] [rbp-50h] BYREF

  v34 = a2;
  v35 = a3;
  v10 = sub_349D6E0(a5, a6);
  v37 = a5;
  v36 = a4;
  v11 = (const __m128i *)sub_349D6E0(a5, a6);
  sub_34A9810(v35, v11, v12, v13, v14, v15);
  if ( a7 == 1 )
  {
    sub_349EA70((__int64)&v31, a1, a2);
    v20 = v32;
    v21 = v33;
    v30 = v31;
    sub_349DE60(v44, v10);
    v22 = v45;
    v23 = &v45[32 * v46];
    if ( v45 == v23 )
      goto LABEL_73;
    v24 = *(_DWORD *)a8;
    while ( 1 )
    {
      if ( *(_DWORD *)v22 != v24 )
        goto LABEL_29;
      if ( v24 == 2 )
      {
        if ( *((_DWORD *)v22 + 2) == *(_DWORD *)(a8 + 8)
          && *((_QWORD *)v22 + 2) == *(_QWORD *)(a8 + 16)
          && *((_QWORD *)v22 + 3) == *(_QWORD *)(a8 + 24) )
        {
          goto LABEL_35;
        }
      }
      else
      {
        if ( v24 <= 2 )
        {
          if ( v24 != 1 )
LABEL_74:
            BUG();
LABEL_40:
          if ( *((_QWORD *)v22 + 1) == *(_QWORD *)(a8 + 8) )
            goto LABEL_35;
          goto LABEL_29;
        }
        if ( v24 == 3 )
          goto LABEL_40;
        if ( v24 != 4 )
          goto LABEL_74;
        if ( *((_DWORD *)v22 + 2) == *(_DWORD *)(a8 + 8) && *((_QWORD *)v22 + 2) == *(_QWORD *)(a8 + 16) )
        {
LABEL_35:
          *(_DWORD *)v22 = 2;
          *((_DWORD *)v22 + 2) = v30;
          *((_QWORD *)v22 + 2) = v20;
          *((_QWORD *)v22 + 3) = v21;
          sub_34AC2B0(&v34, v44);
          if ( v48 != &v49 )
            _libc_free((unsigned __int64)v48);
          v19 = (unsigned __int64)v45;
          if ( v45 == &v47 )
            return;
LABEL_21:
          _libc_free(v19);
          return;
        }
      }
LABEL_29:
      v22 += 32;
      if ( v23 == v22 )
        goto LABEL_73;
    }
  }
  if ( a7 == 2 )
  {
    sub_349DE60(v50, v10);
    v25 = v51;
    v26 = &v51[32 * v52];
    if ( v51 != v26 )
    {
      v27 = *(_DWORD *)a8;
      do
      {
        if ( *(_DWORD *)v25 != v27 )
          goto LABEL_48;
        if ( v27 == 2 )
        {
          if ( *((_DWORD *)v25 + 2) == *(_DWORD *)(a8 + 8)
            && *((_QWORD *)v25 + 2) == *(_QWORD *)(a8 + 16)
            && *((_QWORD *)v25 + 3) == *(_QWORD *)(a8 + 24) )
          {
            goto LABEL_54;
          }
        }
        else
        {
          if ( v27 <= 2 )
          {
            if ( v27 != 1 )
              goto LABEL_74;
LABEL_59:
            if ( *((_QWORD *)v25 + 1) == *(_QWORD *)(a8 + 8) )
              goto LABEL_54;
            goto LABEL_48;
          }
          if ( v27 == 3 )
            goto LABEL_59;
          if ( v27 != 4 )
            goto LABEL_74;
          if ( *((_DWORD *)v25 + 2) == *(_DWORD *)(a8 + 8) && *((_QWORD *)v25 + 2) == *(_QWORD *)(a8 + 16) )
          {
LABEL_54:
            *(_DWORD *)v25 = 1;
            *((_QWORD *)v25 + 1) = a9;
            sub_34AC2B0(&v34, v50);
            if ( v54 != &v55 )
              _libc_free((unsigned __int64)v54);
            v19 = (unsigned __int64)v51;
            if ( v51 == &v53 )
              return;
            goto LABEL_21;
          }
        }
LABEL_48:
        v25 += 32;
      }
      while ( v26 != v25 );
    }
LABEL_73:
    BUG();
  }
  sub_349DE60(v38, v10);
  v16 = v39;
  v17 = &v39[32 * v40];
  if ( v39 == v17 )
    goto LABEL_73;
  v18 = *(_DWORD *)a8;
  while ( 1 )
  {
    if ( *(_DWORD *)v16 != v18 )
      goto LABEL_9;
    if ( v18 != 2 )
      break;
    if ( *((_DWORD *)v16 + 2) == *(_DWORD *)(a8 + 8)
      && *((_QWORD *)v16 + 2) == *(_QWORD *)(a8 + 16)
      && *((_QWORD *)v16 + 3) == *(_QWORD *)(a8 + 24) )
    {
      goto LABEL_18;
    }
LABEL_9:
    v16 += 32;
    if ( v17 == v16 )
      goto LABEL_73;
  }
  if ( v18 <= 2 )
  {
    if ( v18 != 1 )
      goto LABEL_74;
LABEL_17:
    if ( *((_QWORD *)v16 + 1) == *(_QWORD *)(a8 + 8) )
      goto LABEL_18;
    goto LABEL_9;
  }
  if ( v18 == 3 )
    goto LABEL_17;
  if ( v18 != 4 )
    goto LABEL_74;
  if ( *((_DWORD *)v16 + 2) != *(_DWORD *)(a8 + 8) || *((_QWORD *)v16 + 2) != *(_QWORD *)(a8 + 16) )
    goto LABEL_9;
LABEL_18:
  *(_DWORD *)v16 = 1;
  *((_QWORD *)v16 + 1) = a9;
  sub_34AC2B0(&v34, v38);
  if ( v42 != &v43 )
    _libc_free((unsigned __int64)v42);
  v19 = (unsigned __int64)v39;
  if ( v39 != &v41 )
    goto LABEL_21;
}
