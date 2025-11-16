// Function: sub_15B8340
// Address: 0x15b8340
//
__int64 __fastcall sub_15B8340(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // r14d
  __int64 v6; // rax
  __int64 v7; // r12
  unsigned int v9; // esi
  __int64 v10; // rdi
  __int64 v11; // r8
  __int64 v12; // rcx
  char v13; // cl
  __int64 v14; // r10
  int v15; // r10d
  __int64 *v16; // r10
  __int64 *v17; // r11
  __int64 v18; // rax
  int v19; // edx
  unsigned int v20; // edx
  _QWORD *v21; // rdi
  __int64 v22; // rcx
  __int64 v23; // rsi
  _QWORD *v24; // r15
  __int64 v25; // rax
  __int64 v26; // r8
  __int64 v27; // rax
  __int64 v28; // r10
  __int64 v29; // rax
  __int64 v30; // rax
  int v31; // [rsp+4h] [rbp-BCh]
  __int64 v32; // [rsp+8h] [rbp-B8h]
  __int64 v33; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v34; // [rsp+18h] [rbp-A8h] BYREF
  __int64 v35; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v36; // [rsp+28h] [rbp-98h] BYREF
  int v37; // [rsp+30h] [rbp-90h] BYREF
  __int64 v38; // [rsp+38h] [rbp-88h] BYREF
  bool v39; // [rsp+40h] [rbp-80h]
  bool v40; // [rsp+41h] [rbp-7Fh]
  int v41; // [rsp+44h] [rbp-7Ch]
  __int64 v42; // [rsp+48h] [rbp-78h]
  int v43; // [rsp+50h] [rbp-70h]
  int v44; // [rsp+54h] [rbp-6Ch]
  int v45; // [rsp+58h] [rbp-68h]
  int v46; // [rsp+5Ch] [rbp-64h]
  bool v47; // [rsp+60h] [rbp-60h]
  __int64 v48; // [rsp+68h] [rbp-58h]
  __int64 v49; // [rsp+70h] [rbp-50h]
  __int64 v50; // [rsp+78h] [rbp-48h]
  __int64 v51; // [rsp+80h] [rbp-40h]
  __int64 v52; // [rsp+88h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v7 = *(_QWORD *)(a1 + 8);
    v9 = *(_DWORD *)(*a2 + 8);
    v10 = *(_QWORD *)(v6 + 8 * (1LL - v9));
    v33 = v10;
    v34 = *(_QWORD *)(v6 + 8 * (2LL - v9));
    v11 = *(_QWORD *)(v6 + 8 * (3LL - v9));
    v12 = v6;
    v35 = v11;
    if ( *(_BYTE *)v6 != 15 )
      v12 = *(_QWORD *)(v6 - 8LL * *(unsigned int *)(v6 + 8));
    v36 = v12;
    v37 = *(_DWORD *)(v6 + 24);
    v38 = *(_QWORD *)(v6 + 8 * (4LL - v9));
    v13 = *(_BYTE *)(v6 + 40);
    v39 = (v13 & 4) != 0;
    v40 = (v13 & 8) != 0;
    v41 = *(_DWORD *)(v6 + 28);
    v14 = 0;
    if ( v9 > 8 )
      v14 = *(_QWORD *)(v6 + 8 * (8LL - v9));
    v42 = v14;
    v43 = *(_BYTE *)(v6 + 40) & 3;
    v44 = *(_DWORD *)(v6 + 32);
    v45 = *(_DWORD *)(v6 + 36);
    v15 = *(_DWORD *)(v6 + 44);
    v47 = (v13 & 0x10) != 0;
    v46 = v15;
    v48 = *(_QWORD *)(v6 + 8 * (5LL - v9));
    v16 = (__int64 *)(v6 + 8 * (6LL - v9));
    v17 = (__int64 *)(v6 + 8 * (7LL - v9));
    if ( v9 <= 9 )
    {
      v49 = 0;
      v50 = *v16;
      v51 = *v17;
      v18 = 0;
    }
    else
    {
      v49 = *(_QWORD *)(v6 + 8 * (9LL - v9));
      v50 = *v16;
      v51 = *v17;
      if ( v9 == 10 )
        v18 = 0;
      else
        v18 = *(_QWORD *)(v6 + 8 * (10LL - v9));
    }
    v52 = v18;
    if ( v11 != 0
      && v10 != 0
      && (v13 & 8) == 0
      && *(_BYTE *)v10 == 13
      && *(_QWORD *)(v10 + 8 * (7LL - *(unsigned int *)(v10 + 8))) )
    {
      v19 = sub_15B2D00(&v35, &v33);
    }
    else
    {
      v19 = sub_15B55D0(&v34, &v33, &v36, &v38, &v37);
    }
    v20 = (v4 - 1) & v19;
    v21 = (_QWORD *)(v7 + 8LL * v20);
    v22 = *a2;
    v23 = *v21;
    if ( *v21 == *a2 )
    {
LABEL_36:
      *a3 = v21;
      return 1;
    }
    else
    {
      v31 = 1;
      v24 = 0;
      while ( v23 != -8 )
      {
        if ( v23 == -16 )
        {
          if ( !v24 )
            v24 = v21;
        }
        else
        {
          v32 = 0;
          v25 = *(unsigned int *)(v22 + 8);
          if ( (unsigned int)v25 > 9 )
            v32 = *(_QWORD *)(v22 + 8 * (9 - v25));
          v26 = *(_QWORD *)(v22 + 8 * (3 - v25));
          if ( v26 )
          {
            v27 = *(_QWORD *)(v22 + 8 * (1 - v25));
            if ( (*(_BYTE *)(v22 + 40) & 8) == 0 )
            {
              if ( v27 )
              {
                if ( *(_BYTE *)v27 == 13 )
                {
                  if ( *(_QWORD *)(v27 + 8 * (7LL - *(unsigned int *)(v27 + 8))) )
                  {
                    if ( (*(_BYTE *)(v23 + 40) & 8) == 0 )
                    {
                      v28 = *(unsigned int *)(v23 + 8);
                      if ( *(_QWORD *)(v23 + 8 * (1 - v28)) == v27 )
                      {
                        v29 = *(_QWORD *)(v23 + 8 * (3 - v28));
                        if ( v26 == v29 )
                        {
                          if ( v29 )
                          {
                            v30 = 0;
                            if ( *(_DWORD *)(v23 + 8) > 9u )
                              v30 = *(_QWORD *)(v23 + 8 * (9 - v28));
                            if ( v32 == v30 )
                              goto LABEL_36;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        v20 = (v4 - 1) & (v31 + v20);
        v21 = (_QWORD *)(v7 + 8LL * v20);
        v23 = *v21;
        if ( *v21 == v22 )
          goto LABEL_36;
        ++v31;
      }
      if ( !v24 )
        v24 = v21;
      *a3 = v24;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
