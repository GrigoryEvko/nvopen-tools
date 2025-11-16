// Function: sub_1F79A30
// Address: 0x1f79a30
//
__int64 __fastcall sub_1F79A30(
        __int64 a1,
        unsigned int a2,
        unsigned __int8 a3,
        __int64 a4,
        _BYTE *a5,
        unsigned int a6,
        double a7,
        double a8,
        double a9)
{
  unsigned __int8 *v9; // rax
  __int64 v10; // r13
  unsigned int v11; // ebx
  unsigned int v12; // r15d
  _BOOL4 v13; // r12d
  __int64 result; // rax
  __int64 (*v15)(); // rax
  __int16 v16; // cx
  unsigned __int16 v17; // cx
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned int v20; // ebx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 (*v23)(); // r12
  __int64 *v24; // rsi
  void *v25; // rbx
  __int64 v26; // rdx
  __int64 v27; // r13
  __int64 v28; // r12
  __int64 v29; // r14
  __int64 v30; // rax
  __int64 v31; // r15
  __int64 v32; // r13
  __int64 v33; // rdx
  __int64 v34; // r12
  __int64 v35; // r14
  __int64 v36; // r13
  __int64 v37; // rdx
  __int64 v38; // r15
  __int64 v39; // r12
  __int64 v40; // rcx
  __int64 v41; // [rsp+0h] [rbp-90h]
  unsigned __int8 v42; // [rsp+Fh] [rbp-81h]
  char v43; // [rsp+Fh] [rbp-81h]
  char v44; // [rsp+Fh] [rbp-81h]
  unsigned __int8 v45; // [rsp+10h] [rbp-80h]
  __int64 v46; // [rsp+10h] [rbp-80h]
  __int64 v47; // [rsp+10h] [rbp-80h]
  _BYTE *v48; // [rsp+18h] [rbp-78h]
  unsigned __int8 v49; // [rsp+18h] [rbp-78h]
  unsigned __int8 v50; // [rsp+18h] [rbp-78h]
  __int64 v51; // [rsp+18h] [rbp-78h]
  __int64 v52; // [rsp+18h] [rbp-78h]
  unsigned __int8 v53; // [rsp+18h] [rbp-78h]
  unsigned __int8 v54; // [rsp+18h] [rbp-78h]
  void *v55; // [rsp+28h] [rbp-68h] BYREF
  __int64 v56; // [rsp+30h] [rbp-60h]
  _BYTE v57[8]; // [rsp+40h] [rbp-50h] BYREF
  void *v58; // [rsp+48h] [rbp-48h] BYREF
  __int64 v59; // [rsp+50h] [rbp-40h]

  while ( 1 )
  {
    while ( 1 )
    {
      v45 = a3;
      v48 = a5;
      if ( *(_WORD *)(a1 + 24) == 162 )
        return 2;
      v9 = (unsigned __int8 *)(*(_QWORD *)(a1 + 40) + 16LL * a2);
      v10 = a4;
      v11 = a6;
      v42 = *v9;
      v12 = *v9;
      v13 = (*(_BYTE *)(a1 + 80) & 0x40) != 0;
      v41 = *((_QWORD *)v9 + 1);
      if ( !sub_1D18C00(a1, 1, a2) )
      {
        if ( *(_WORD *)(a1 + 24) != 157 )
          return 0;
        v15 = *(__int64 (**)())(*(_QWORD *)v10 + 880LL);
        if ( v15 == sub_1D5A410
          || !((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, _QWORD))v15)(
                v10,
                v12,
                v41,
                *(unsigned __int8 *)(*(_QWORD *)(**(_QWORD **)(a1 + 32) + 40LL)
                                   + 16LL * *(unsigned int *)(*(_QWORD *)(a1 + 32) + 8LL)),
                *(_QWORD *)(*(_QWORD *)(**(_QWORD **)(a1 + 32) + 40LL)
                          + 16LL * *(unsigned int *)(*(_QWORD *)(a1 + 32) + 8LL)
                          + 8)) )
        {
          return 0;
        }
      }
      if ( v11 > 6 )
        return 0;
      v16 = *(_WORD *)(a1 + 24);
      if ( v16 == 77 )
        return v13 | (unsigned int)((*v48 & 0x20) != 0);
      if ( v16 <= 77 )
        break;
      if ( v16 <= 79 )
        goto LABEL_24;
      v17 = v16 - 154;
      if ( v17 > 0xBu || ((1LL << v17) & 0x809) == 0 )
        return 0;
      v18 = *(_QWORD *)(a1 + 32);
      a3 = v45;
      a6 = v11 + 1;
      a4 = v10;
      a5 = v48;
      a2 = *(_DWORD *)(v18 + 8);
      a1 = *(_QWORD *)v18;
    }
    if ( v16 == 11 )
      break;
    if ( v16 != 76 || (*v48 & 2) == 0 && !v13 )
      return 0;
    if ( v45 )
    {
      v19 = 1;
      if ( v42 != 1 )
      {
        if ( !v42 )
          return 0;
        v19 = v42;
        if ( !*(_QWORD *)(v10 + 8LL * v42 + 120) )
          return 0;
      }
      if ( (*(_BYTE *)(v10 + 259 * v19 + 2499) & 0xFB) != 0 )
        return 0;
    }
LABEL_24:
    v20 = v11 + 1;
    result = sub_1F79A30(**(_QWORD **)(a1 + 32), *(unsigned int *)(*(_QWORD *)(a1 + 32) + 8LL), v45, v10, v48, v20);
    if ( (_BYTE)result )
      return result;
    v21 = *(_QWORD *)(a1 + 32);
    a6 = v20;
    a5 = v48;
    a4 = v10;
    a3 = v45;
    a2 = *(_DWORD *)(v21 + 48);
    a1 = *(_QWORD *)(v21 + 40);
  }
  result = 1;
  if ( v45 )
  {
    if ( (v22 = 1, v42 != 1) && (!v42 || (v22 = v42, !*(_QWORD *)(v10 + 8LL * v42 + 120)))
      || (v26 = 259 * v22, result = 1, *(_BYTE *)(v10 + v26 + 2433)) )
    {
      v23 = *(__int64 (**)())(*(_QWORD *)v10 + 328LL);
      v24 = (__int64 *)(*(_QWORD *)(a1 + 88) + 32LL);
      v25 = sub_16982C0();
      if ( (void *)*v24 == v25 )
        sub_169C6E0(&v55, (__int64)v24);
      else
        sub_16986C0(&v55, v24);
      if ( v55 == v25 )
        sub_169C8D0((__int64)&v55, a7, a8, a9);
      else
        sub_1699490((__int64)&v55);
      if ( v55 == v25 )
        sub_169C7E0(&v58, &v55);
      else
        sub_1698450((__int64)&v58, (__int64)&v55);
      result = 0;
      if ( v23 != sub_1F3CA70 )
        result = ((__int64 (__fastcall *)(__int64, _BYTE *, _QWORD, __int64))v23)(v10, v57, v12, v41);
      if ( v25 == v58 )
      {
        v34 = v59;
        if ( v59 )
        {
          if ( v59 != v59 + 32LL * *(_QWORD *)(v59 - 8) )
          {
            v35 = v59 + 32LL * *(_QWORD *)(v59 - 8);
            v47 = v59;
            v44 = result;
            do
            {
              v35 -= 32;
              if ( v25 == *(void **)(v35 + 8) )
              {
                v36 = *(_QWORD *)(v35 + 16);
                if ( v36 )
                {
                  v37 = 32LL * *(_QWORD *)(v36 - 8);
                  v38 = v36 + v37;
                  while ( v36 != v38 )
                  {
                    v38 -= 32;
                    if ( v25 == *(void **)(v38 + 8) )
                    {
                      v39 = *(_QWORD *)(v38 + 16);
                      if ( v39 )
                      {
                        v40 = v39 + 32LL * *(_QWORD *)(v39 - 8);
                        if ( v39 != v40 )
                        {
                          do
                          {
                            v52 = v40 - 32;
                            sub_127D120((_QWORD *)(v40 - 24));
                            v40 = v52;
                          }
                          while ( v39 != v52 );
                        }
                        j_j_j___libc_free_0_0(v39 - 8);
                      }
                    }
                    else
                    {
                      sub_1698460(v38 + 8);
                    }
                  }
                  j_j_j___libc_free_0_0(v36 - 8);
                }
              }
              else
              {
                sub_1698460(v35 + 8);
              }
            }
            while ( v47 != v35 );
            v34 = v47;
            LOBYTE(result) = v44;
          }
          v54 = result;
          j_j_j___libc_free_0_0(v34 - 8);
          result = v54;
        }
      }
      else
      {
        v49 = result;
        sub_1698460((__int64)&v58);
        result = v49;
      }
      if ( v25 == v55 )
      {
        v27 = v56;
        if ( v56 )
        {
          v28 = v56 + 32LL * *(_QWORD *)(v56 - 8);
          if ( v56 != v28 )
          {
            v46 = v56;
            v43 = result;
            do
            {
              v28 -= 32;
              if ( v25 == *(void **)(v28 + 8) )
              {
                v29 = *(_QWORD *)(v28 + 16);
                if ( v29 )
                {
                  v30 = 32LL * *(_QWORD *)(v29 - 8);
                  v31 = v29 + v30;
                  while ( v29 != v31 )
                  {
                    v31 -= 32;
                    if ( v25 == *(void **)(v31 + 8) )
                    {
                      v32 = *(_QWORD *)(v31 + 16);
                      if ( v32 )
                      {
                        v33 = v32 + 32LL * *(_QWORD *)(v32 - 8);
                        if ( v32 != v33 )
                        {
                          do
                          {
                            v51 = v33 - 32;
                            sub_127D120((_QWORD *)(v33 - 24));
                            v33 = v51;
                          }
                          while ( v32 != v51 );
                        }
                        j_j_j___libc_free_0_0(v32 - 8);
                      }
                    }
                    else
                    {
                      sub_1698460(v31 + 8);
                    }
                  }
                  j_j_j___libc_free_0_0(v29 - 8);
                }
              }
              else
              {
                sub_1698460(v28 + 8);
              }
            }
            while ( v46 != v28 );
            v27 = v46;
            LOBYTE(result) = v43;
          }
          v53 = result;
          j_j_j___libc_free_0_0(v27 - 8);
          return v53;
        }
      }
      else
      {
        v50 = result;
        sub_1698460((__int64)&v55);
        return v50;
      }
    }
  }
  return result;
}
