// Function: sub_390E4A0
// Address: 0x390e4a0
//
double __fastcall sub_390E4A0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 **v4; // r15
  __int64 *v5; // r8
  __int64 v6; // rax
  __int64 *v9; // r13
  __int64 v10; // r14
  int v11; // r8d
  __int64 **v12; // r9
  unsigned int v13; // edx
  __int64 **v14; // r15
  __int64 **v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 **v18; // r13
  __int64 **v19; // r14
  double v20; // xmm0_8
  __int64 **v21; // r15
  double v22; // xmm1_8
  __int64 **v23; // rsi
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 **v31; // r13
  __int64 **v32; // rbx
  __int64 **v33; // r12
  __int64 v34; // rcx
  char **v35; // rsi
  __int64 v36; // rdi
  unsigned __int64 *v37; // rbx
  __int64 **v38; // r14
  _QWORD *v39; // [rsp+8h] [rbp-358h]
  __int64 v40; // [rsp+10h] [rbp-350h]
  unsigned int v41; // [rsp+18h] [rbp-348h]
  __int64 *v42; // [rsp+18h] [rbp-348h]
  __int64 v43; // [rsp+18h] [rbp-348h]
  __int64 v44; // [rsp+30h] [rbp-330h]
  int v45; // [rsp+30h] [rbp-330h]
  double v46; // [rsp+38h] [rbp-328h]
  __int64 *v48; // [rsp+48h] [rbp-318h]
  char *v49; // [rsp+50h] [rbp-310h] BYREF
  __int64 v50; // [rsp+58h] [rbp-308h]
  _BYTE v51[64]; // [rsp+60h] [rbp-300h] BYREF
  __int64 **v52; // [rsp+A0h] [rbp-2C0h]
  __int64 v53; // [rsp+A8h] [rbp-2B8h]
  _BYTE v54[688]; // [rsp+B0h] [rbp-2B0h] BYREF

  v4 = (__int64 **)v54;
  v53 = 0x800000000LL;
  v5 = *(__int64 **)a2;
  v6 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  v52 = (__int64 **)v54;
  v48 = (__int64 *)v6;
  if ( v5 == (__int64 *)v6 )
    return 0.0;
  v9 = v5;
  do
  {
    v10 = *v9;
    if ( (*(_QWORD *)(a1 + 8) & *(_QWORD *)(*v9 + 48)) == 0 )
      goto LABEL_19;
    v44 = sub_390E170(a1, *v9, a3, a4);
    v13 = v53;
    if ( v4 == &v52[10 * (unsigned int)v53] )
      goto LABEL_7;
    if ( sub_390E170(a1, **v4, a3, a4) != v44 )
    {
      v13 = v53;
LABEL_7:
      v49 = v51;
      v50 = 0x800000000LL;
      if ( HIDWORD(v53) <= v13 )
      {
        v41 = v13;
        v25 = (((((unsigned __int64)HIDWORD(v53) + 2) >> 1) | (HIDWORD(v53) + 2LL)) >> 2)
            | (((unsigned __int64)HIDWORD(v53) + 2) >> 1)
            | (HIDWORD(v53) + 2LL);
        v26 = (((v25 >> 4) | v25) >> 8) | (v25 >> 4) | v25;
        v27 = (v26 | (v26 >> 16) | HIDWORD(v26)) + 1;
        v28 = 0xFFFFFFFFLL;
        if ( v27 <= 0xFFFFFFFF )
          v28 = v27;
        v45 = v28;
        v29 = malloc(80 * v28);
        v30 = v41;
        v14 = (__int64 **)v29;
        if ( !v29 )
        {
          sub_16BD1C0("Allocation failed", 1u);
          v30 = (unsigned int)v53;
        }
        v12 = &v52[10 * v30];
        if ( v52 != v12 )
        {
          v42 = v9;
          v31 = v52;
          v40 = a1;
          v32 = &v52[10 * v30];
          v39 = a4;
          v33 = v14;
          do
          {
            while ( 1 )
            {
              if ( v33 )
              {
                *((_DWORD *)v33 + 2) = 0;
                *v33 = (__int64 *)(v33 + 2);
                *((_DWORD *)v33 + 3) = 8;
                v34 = *((unsigned int *)v31 + 2);
                if ( (_DWORD)v34 )
                  break;
              }
              v31 += 10;
              v33 += 10;
              if ( v32 == v31 )
                goto LABEL_42;
            }
            v35 = (char **)v31;
            v36 = (__int64)v33;
            v31 += 10;
            v33 += 10;
            sub_390DA00(v36, v35, v30, v34, v11, (int)v12);
          }
          while ( v32 != v31 );
LABEL_42:
          v9 = v42;
          a1 = v40;
          a4 = v39;
          v12 = &v52[10 * (unsigned int)v53];
          if ( v52 != v12 )
          {
            v43 = v10;
            v37 = (unsigned __int64 *)&v52[10 * (unsigned int)v53];
            v38 = v52;
            do
            {
              v37 -= 10;
              if ( (unsigned __int64 *)*v37 != v37 + 2 )
                _libc_free(*v37);
            }
            while ( v37 != (unsigned __int64 *)v38 );
            v10 = v43;
            a1 = v40;
            v12 = v52;
          }
        }
        if ( v12 != (__int64 **)v54 )
          _libc_free((unsigned __int64)v12);
        v52 = v14;
        v13 = v53;
        HIDWORD(v53) = v45;
      }
      else
      {
        v14 = v52;
      }
      v15 = &v14[10 * v13];
      if ( v15 )
      {
        *v15 = (__int64 *)(v15 + 2);
        v15[1] = (__int64 *)0x800000000LL;
        if ( (_DWORD)v50 )
          sub_390DA00((__int64)v15, &v49, (unsigned int)v50, v13, v11, (int)v12);
        v13 = v53;
      }
      v16 = v13 + 1;
      LODWORD(v53) = v16;
      if ( v49 != v51 )
      {
        _libc_free((unsigned __int64)v49);
        v16 = (unsigned int)v53;
      }
      v4 = &v52[10 * v16 - 10];
    }
    v17 = *((unsigned int *)v4 + 2);
    if ( (unsigned int)v17 >= *((_DWORD *)v4 + 3) )
    {
      sub_16CD150((__int64)v4, v4 + 2, 0, 8, v11, (int)v12);
      v17 = *((unsigned int *)v4 + 2);
    }
    (*v4)[v17] = v10;
    ++*((_DWORD *)v4 + 2);
LABEL_19:
    ++v9;
  }
  while ( v48 != v9 );
  v18 = v52;
  v46 = 0.0;
  if ( (_DWORD)v53 )
  {
    v19 = v52 + 10;
    v20 = sub_390E1B0(a1, v52, a3, a4);
    v21 = v52;
    v22 = 0.0;
    v18 = &v52[10 * (unsigned int)v53];
    v46 = v20 + 0.0;
    if ( v19 != v18 )
    {
      do
      {
        v23 = v19;
        v19 += 10;
        v22 = v22 + (**(double (__fastcall ***)(__int64, __int64 **, __int64, _QWORD *))a1)(a1, v23, a3, a4);
      }
      while ( v18 != v19 );
      v21 = v52;
      v46 = v22 + v46;
      v18 = &v52[10 * (unsigned int)v53];
    }
    if ( v21 != v18 )
    {
      do
      {
        v18 -= 10;
        if ( *v18 != (__int64 *)(v18 + 2) )
          _libc_free((unsigned __int64)*v18);
      }
      while ( v18 != v21 );
      v18 = v52;
    }
  }
  if ( v18 != (__int64 **)v54 )
    _libc_free((unsigned __int64)v18);
  return v46;
}
