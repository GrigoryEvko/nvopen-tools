// Function: sub_10C83A0
// Address: 0x10c83a0
//
__int64 __fastcall sub_10C83A0(unsigned __int8 *a1, __int64 *a2)
{
  int v2; // r14d
  unsigned __int8 *v3; // rbx
  __int64 v4; // r11
  __int64 v5; // rax
  unsigned int v6; // r13d
  __int64 v7; // rax
  char v9; // al
  __int64 v10; // rax
  char v11; // al
  char v12; // al
  __int64 v13; // rsi
  __int64 v14; // r14
  __int64 v15; // rbx
  __int64 v16; // r12
  __int64 v17; // r13
  __int64 v18; // rbx
  __int64 v19; // r14
  char v20; // al
  __int64 v21; // rsi
  __int64 v22; // rax
  char v23; // al
  char v24; // dl
  __int64 v25; // rax
  __int64 v26; // rax
  char v27; // dl
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 *v30; // r14
  const char *v31; // rax
  __int64 *v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rbx
  __int64 v36; // r12
  __int64 v37; // rdx
  unsigned int v38; // esi
  __int64 v39; // rdx
  int v40; // r13d
  __int64 v41; // r13
  __int64 v42; // rbx
  __int64 v43; // rdx
  unsigned int v44; // esi
  char v45; // al
  __int64 v46; // rax
  __int64 v47; // [rsp+18h] [rbp-F8h]
  __int64 v48; // [rsp+18h] [rbp-F8h]
  __int64 v49; // [rsp+18h] [rbp-F8h]
  __int64 v50; // [rsp+18h] [rbp-F8h]
  __int64 v51; // [rsp+18h] [rbp-F8h]
  __int64 v52; // [rsp+18h] [rbp-F8h]
  int v53; // [rsp+24h] [rbp-ECh]
  __int64 v55; // [rsp+38h] [rbp-D8h] BYREF
  __int64 v56; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v57; // [rsp+48h] [rbp-C8h] BYREF
  _QWORD v58[4]; // [rsp+50h] [rbp-C0h] BYREF
  __int16 v59; // [rsp+70h] [rbp-A0h]
  _QWORD *v60; // [rsp+80h] [rbp-90h] BYREF
  __int64 *v61; // [rsp+88h] [rbp-88h]
  __int16 v62; // [rsp+A0h] [rbp-70h]
  __int64 *v63; // [rsp+B0h] [rbp-60h] BYREF
  __int64 *v64; // [rsp+B8h] [rbp-58h] BYREF
  __int64 *v65; // [rsp+C0h] [rbp-50h]
  int v66; // [rsp+C8h] [rbp-48h]
  __int16 v67; // [rsp+D0h] [rbp-40h]

  v2 = *a1;
  v3 = (unsigned __int8 *)*((_QWORD *)a1 - 8);
  v60 = 0;
  v61 = &v55;
  v4 = *((_QWORD *)a1 - 4);
  v53 = v2 - 29;
  v5 = *((_QWORD *)v3 + 2);
  v6 = (v2 == 57) + 28;
  if ( v5 && !*(_QWORD *)(v5 + 8) && *v3 == 59 )
  {
    v51 = *((_QWORD *)a1 - 4);
    v20 = sub_995B10(&v60, *((_QWORD *)v3 - 8));
    v21 = *((_QWORD *)v3 - 4);
    v4 = v51;
    if ( v20 && v21 )
    {
      *v61 = v21;
    }
    else
    {
      v45 = sub_995B10(&v60, v21);
      v4 = v51;
      if ( !v45 )
        goto LABEL_2;
      v46 = *((_QWORD *)v3 - 8);
      if ( !v46 )
        goto LABEL_2;
      *v61 = v46;
    }
    v63 = 0;
    v64 = &v56;
    v22 = *(_QWORD *)(v4 + 16);
    if ( v22 )
    {
      if ( !*(_QWORD *)(v22 + 8) )
      {
        v52 = v4;
        v23 = sub_996420(&v63, 30, (unsigned __int8 *)v4);
        v4 = v52;
        if ( v23 )
        {
          v24 = 0;
          v25 = *(_QWORD *)(v55 + 16);
          if ( v25 )
            v24 = *(_QWORD *)(v25 + 8) == 0;
          LOBYTE(v58[0]) = 0;
          v26 = sub_F13D80(a2, v55, v24, 0, v58, 0);
          v4 = v52;
          if ( !v26 )
          {
            v27 = 0;
            v28 = *(_QWORD *)(v56 + 16);
            if ( v28 )
              v27 = *(_QWORD *)(v28 + 8) == 0;
            LOBYTE(v58[0]) = 0;
            v29 = sub_F13D80(a2, v56, v27, 0, v58, 0);
            v4 = v52;
            if ( !v29 )
            {
              v30 = (__int64 *)a2[4];
              v31 = sub_BD5D20((__int64)a1);
              v67 = 773;
              v64 = v32;
              v63 = (__int64 *)v31;
              v65 = (__int64 *)".demorgan";
              v33 = sub_10BBE20(v30, v6, v55, v56, (int)v60, 0, (__int64)&v63, 0);
              v67 = 257;
              return sub_B50640(v33, (__int64)&v63, 0, 0);
            }
          }
        }
      }
    }
  }
LABEL_2:
  v63 = &v55;
  v64 = 0;
  v65 = &v56;
  v66 = v2 - 29;
  v7 = *((_QWORD *)v3 + 2);
  if ( !v7 || *(_QWORD *)(v7 + 8) || v2 != *v3 )
    return 0;
  if ( !*((_QWORD *)v3 - 8)
    || (v55 = *((_QWORD *)v3 - 8), v47 = v4, v9 = sub_996420(&v64, 30, *((unsigned __int8 **)v3 - 4)), v4 = v47, !v9) )
  {
    v10 = *((_QWORD *)v3 - 4);
    if ( !v10 )
      return 0;
    v48 = v4;
    *v63 = v10;
    v11 = sub_996420(&v64, 30, *((unsigned __int8 **)v3 - 8));
    v4 = v48;
    if ( !v11 )
      return 0;
  }
  v60 = 0;
  v61 = &v57;
  if ( *(_BYTE *)v4 != 59 )
    return 0;
  v49 = v4;
  v12 = sub_995B10(&v60, *(_QWORD *)(v4 - 64));
  v13 = *(_QWORD *)(v49 - 32);
  if ( !v12 || !v13 )
  {
    if ( (unsigned __int8)sub_995B10(&v60, v13) )
    {
      v34 = *(_QWORD *)(v49 - 64);
      if ( v34 )
      {
        *v61 = v34;
        goto LABEL_14;
      }
    }
    return 0;
  }
  *v61 = v13;
LABEL_14:
  v62 = 257;
  v14 = v56;
  v15 = a2[4];
  v50 = v57;
  v16 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(v15 + 80) + 16LL))(
          *(_QWORD *)(v15 + 80),
          v6,
          v56,
          v57);
  if ( !v16 )
  {
    v67 = 257;
    v16 = sub_B504D0(v6, v14, v50, (__int64)&v63, 0, 0);
    if ( (unsigned __int8)sub_920620(v16) )
    {
      v39 = *(_QWORD *)(v15 + 96);
      v40 = *(_DWORD *)(v15 + 104);
      if ( v39 )
        sub_B99FD0(v16, 3u, v39);
      sub_B45150(v16, v40);
    }
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD **, _QWORD, _QWORD))(**(_QWORD **)(v15 + 88) + 16LL))(
      *(_QWORD *)(v15 + 88),
      v16,
      &v60,
      *(_QWORD *)(v15 + 56),
      *(_QWORD *)(v15 + 64));
    v41 = *(_QWORD *)v15;
    v42 = *(_QWORD *)v15 + 16LL * *(unsigned int *)(v15 + 8);
    while ( v42 != v41 )
    {
      v43 = *(_QWORD *)(v41 + 8);
      v44 = *(_DWORD *)v41;
      v41 += 16;
      sub_B99FD0(v16, v44, v43);
    }
  }
  v62 = 257;
  v59 = 257;
  v17 = a2[4];
  v18 = sub_AD62B0(*(_QWORD *)(v16 + 8));
  v19 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v17 + 80) + 16LL))(
          *(_QWORD *)(v17 + 80),
          30,
          v16,
          v18);
  if ( !v19 )
  {
    v67 = 257;
    v19 = sub_B504D0(30, v16, v18, (__int64)&v63, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v17 + 88) + 16LL))(
      *(_QWORD *)(v17 + 88),
      v19,
      v58,
      *(_QWORD *)(v17 + 56),
      *(_QWORD *)(v17 + 64));
    v35 = *(_QWORD *)v17;
    v36 = *(_QWORD *)v17 + 16LL * *(unsigned int *)(v17 + 8);
    if ( *(_QWORD *)v17 != v36 )
    {
      do
      {
        v37 = *(_QWORD *)(v35 + 8);
        v38 = *(_DWORD *)v35;
        v35 += 16;
        sub_B99FD0(v19, v38, v37);
      }
      while ( v36 != v35 );
    }
  }
  return sub_B504D0(v53, v55, v19, (__int64)&v60, 0, 0);
}
