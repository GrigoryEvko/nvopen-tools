// Function: sub_1F7B290
// Address: 0x1f7b290
//
_QWORD *__fastcall sub_1F7B290(
        __int64 **a1,
        _QWORD *a2,
        __int64 a3,
        int a4,
        int a5,
        int a6,
        double a7,
        double a8,
        __m128i a9)
{
  __int64 *v9; // rax
  __int64 v10; // r12
  __int64 v11; // r13
  __int64 v12; // r14
  unsigned __int64 v13; // r15
  __int64 v14; // rdi
  unsigned __int8 *v15; // rax
  unsigned int v16; // ebx
  __int64 v17; // rdx
  int v18; // ecx
  int v19; // r8d
  int v20; // r9d
  __int64 v21; // rax
  _QWORD *v22; // rcx
  __int64 v23; // r14
  __int64 v24; // rsi
  __int64 v25; // r15
  __int64 *v26; // rax
  void *v27; // rax
  void *v28; // r12
  void *v29; // rcx
  __int64 v30; // rax
  __int64 *v31; // rsi
  _QWORD *v32; // rbx
  int v34; // eax
  char v35; // al
  int v36; // eax
  char v37; // al
  _QWORD *v38; // rcx
  __int64 v39; // r14
  __int64 v40; // r14
  __int64 v41; // rsi
  __int64 v42; // r12
  __int64 v43; // rsi
  __int64 *v44; // r11
  __int128 v45; // [rsp-10h] [rbp-A0h]
  __int64 v46; // [rsp+0h] [rbp-90h]
  const void **v48; // [rsp+18h] [rbp-78h]
  void *v50; // [rsp+20h] [rbp-70h]
  __int64 *v51; // [rsp+20h] [rbp-70h]
  __int64 v52; // [rsp+28h] [rbp-68h]
  __int64 v53; // [rsp+28h] [rbp-68h]
  _QWORD *v54; // [rsp+28h] [rbp-68h]
  __int64 v55; // [rsp+30h] [rbp-60h] BYREF
  int v56; // [rsp+38h] [rbp-58h]
  __int64 v57; // [rsp+40h] [rbp-50h] BYREF
  void *v58; // [rsp+48h] [rbp-48h] BYREF
  __int64 v59; // [rsp+50h] [rbp-40h]

  v9 = (__int64 *)a2[4];
  v11 = v9[1];
  v12 = v9[5];
  v13 = v9[6];
  v46 = *v9;
  v10 = *v9;
  v14 = *v9;
  v15 = (unsigned __int8 *)a2[5];
  v48 = (const void **)*((_QWORD *)v15 + 1);
  v16 = *v15;
  v52 = sub_1D23470(v14, v11, a3, a4, a5, a6);
  v21 = sub_1D23470(v12, v13, v17, v18, v19, v20);
  v22 = a2;
  if ( v52 && v21 )
  {
    v23 = *(_QWORD *)(v21 + 88);
    v24 = a2[9];
    v25 = *(_QWORD *)(v52 + 88);
    v26 = *a1;
    v55 = v24;
    v53 = (__int64)v26;
    if ( v24 )
    {
      sub_1623A60((__int64)&v55, v24, 2);
      v22 = a2;
    }
    v50 = *(void **)(v25 + 32);
    v56 = *((_DWORD *)v22 + 16);
    v27 = sub_16982C0();
    v28 = v27;
    if ( v50 == v27 )
    {
      v29 = *(void **)(v23 + 32);
      if ( (*(_BYTE *)(*(_QWORD *)(v25 + 40) + 26LL) & 7) != 1 )
      {
LABEL_7:
        v30 = v23 + 32;
        if ( v28 == v29 )
          v30 = *(_QWORD *)(v23 + 40) + 8LL;
        if ( (*(_BYTE *)(v30 + 18) & 7) == 1 )
        {
          v31 = (__int64 *)(v25 + 32);
          if ( v50 != v28 )
          {
LABEL_11:
            sub_16986C0(&v58, v31);
LABEL_12:
            v32 = sub_1D36490(v53, (__int64)&v57, (__int64)&v55, v16, v48, 0, a7, a8, a9);
            if ( v58 == v28 )
            {
              v40 = v59;
              if ( v59 )
              {
                v41 = 32LL * *(_QWORD *)(v59 - 8);
                v42 = v59 + v41;
                if ( v59 != v59 + v41 )
                {
                  do
                  {
                    v42 -= 32;
                    sub_127D120((_QWORD *)(v42 + 8));
                  }
                  while ( v40 != v42 );
                }
                j_j_j___libc_free_0_0(v40 - 8);
              }
            }
            else
            {
              sub_1698460((__int64)&v58);
            }
            if ( v55 )
              sub_161E7C0((__int64)&v55, v55);
            return v32;
          }
        }
        else
        {
          v39 = v23 + 24;
          if ( (unsigned int)sub_14A9E40(v25 + 24, v39) )
            v39 = v25 + 24;
          v31 = (__int64 *)(v39 + 8);
          if ( v28 != *(void **)(v39 + 8) )
            goto LABEL_11;
        }
LABEL_19:
        sub_169C6E0(&v58, (__int64)v31);
        goto LABEL_12;
      }
    }
    else
    {
      v29 = *(void **)(v23 + 32);
      if ( (*(_BYTE *)(v25 + 50) & 7) != 1 )
        goto LABEL_7;
    }
    v31 = (__int64 *)(v23 + 32);
    if ( v27 != v29 )
      goto LABEL_11;
    goto LABEL_19;
  }
  v34 = *(unsigned __int16 *)(v46 + 24);
  if ( v34 != 33 && v34 != 11 )
  {
    v35 = sub_1D16930(v46);
    v22 = a2;
    if ( !v35 )
      return 0;
  }
  v36 = *(unsigned __int16 *)(v12 + 24);
  if ( v36 == 33 )
    return 0;
  if ( v36 == 11 )
    return 0;
  v54 = v22;
  v37 = sub_1D16930(v12);
  v38 = v54;
  if ( v37 )
    return 0;
  v43 = v54[9];
  v57 = v43;
  v44 = *a1;
  if ( v43 )
  {
    v51 = *a1;
    sub_1623A60((__int64)&v57, v43, 2);
    v38 = v54;
    v44 = v51;
  }
  *((_QWORD *)&v45 + 1) = v11;
  *(_QWORD *)&v45 = v10;
  LODWORD(v58) = *((_DWORD *)v38 + 16);
  v32 = sub_1D332F0(v44, 181, (__int64)&v57, v16, v48, 0, a7, a8, a9, v12, v13, v45);
  if ( v57 )
    sub_161E7C0((__int64)&v57, v57);
  return v32;
}
