// Function: sub_329D830
// Address: 0x329d830
//
__int64 __fastcall sub_329D830(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  _QWORD *v8; // rax
  int v9; // edx
  __int64 v10; // r14
  __int64 v11; // r14
  __int64 v12; // rax
  unsigned __int64 *v13; // r8
  __int64 v14; // rdi
  int v15; // eax
  char v16; // r15
  __int64 v17; // rax
  unsigned __int64 *v18; // rax
  __int64 v19; // rdi
  int v20; // edx
  char v21; // r15
  _QWORD *v22; // rdx
  __int64 v23; // rax
  unsigned __int64 *v24; // r8
  __int64 v25; // rdi
  int v26; // eax
  char v27; // r15
  __int64 v28; // rax
  unsigned __int64 *v29; // rax
  __int64 v30; // rdi
  int v31; // edx
  char v32; // r15
  unsigned __int64 *v33; // r8
  __int64 v34; // rdi
  int v35; // eax
  char v36; // r14
  __int64 v37; // rax
  int v38; // edx
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  int v42; // eax
  char v43; // r14
  __int64 v44; // rax
  __int64 v45; // rsi
  __int64 v46; // rdx
  __int64 v47; // rdx
  __m128i v48; // [rsp-88h] [rbp-88h]
  __m128i v49; // [rsp-78h] [rbp-78h]
  __m128i v50; // [rsp-68h] [rbp-68h]
  __m128i v51; // [rsp-58h] [rbp-58h]
  unsigned __int64 v52; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v53; // [rsp-40h] [rbp-40h]

  if ( *(_DWORD *)a4 != *(_DWORD *)(a1 + 24) )
    return 0;
  v8 = *(_QWORD **)(a1 + 40);
  v9 = *(_DWORD *)(a4 + 8);
  v10 = *v8;
  if ( v9 == *(_DWORD *)(*v8 + 24LL) )
  {
    v12 = *(_QWORD *)(a4 + 16);
    v51 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v10 + 40));
    *(_QWORD *)v12 = v51.m128i_i64[0];
    *(_DWORD *)(v12 + 8) = v51.m128i_i32[2];
    v13 = *(unsigned __int64 **)(a4 + 24);
    v14 = *(_QWORD *)(*(_QWORD *)(v10 + 40) + 40LL);
    if ( v14 && ((v15 = *(_DWORD *)(v14 + 24), v15 == 11) || v15 == 35) )
    {
      if ( v13 )
      {
        v41 = *(_QWORD *)(v14 + 96);
        if ( *((_DWORD *)v13 + 2) <= 0x40u && *(_DWORD *)(v41 + 32) <= 0x40u )
        {
          *v13 = *(_QWORD *)(v41 + 24);
          *((_DWORD *)v13 + 2) = *(_DWORD *)(v41 + 32);
          v22 = *(_QWORD **)(a1 + 40);
          goto LABEL_75;
        }
        sub_C43990(*(_QWORD *)(a4 + 24), v41 + 24);
      }
    }
    else
    {
      v53 = 1;
      if ( !v13 )
        v13 = &v52;
      v52 = 0;
      v16 = sub_33D1410(v14, v13);
      if ( v53 > 0x40 && v52 )
        j_j___libc_free_0_0(v52);
      if ( !v16 )
      {
        v17 = *(_QWORD *)(a4 + 16);
        v50 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v10 + 40) + 40LL));
        *(_QWORD *)v17 = v50.m128i_i64[0];
        *(_DWORD *)(v17 + 8) = v50.m128i_i32[2];
        v18 = *(unsigned __int64 **)(a4 + 24);
        v19 = **(_QWORD **)(v10 + 40);
        if ( !v19 || (v20 = *(_DWORD *)(v19 + 24), v20 != 11) && v20 != 35 )
        {
          v53 = 1;
          if ( !v18 )
            v18 = &v52;
          v52 = 0;
          v21 = sub_33D1410(v19, v18);
          if ( v53 > 0x40 && v52 )
            j_j___libc_free_0_0(v52);
          v8 = *(_QWORD **)(a1 + 40);
          v22 = v8;
          if ( !v21 )
          {
LABEL_24:
            v9 = *(_DWORD *)(a4 + 8);
            goto LABEL_4;
          }
LABEL_75:
          v8 = v22;
          if ( !*(_BYTE *)(a4 + 36) || *(_DWORD *)(a4 + 32) == (*(_DWORD *)(a4 + 32) & *(_DWORD *)(v10 + 28)) )
          {
            v34 = v22[5];
            v33 = *(unsigned __int64 **)(a4 + 40);
            if ( v34 )
            {
              v42 = *(_DWORD *)(v34 + 24);
              if ( v42 == 35 || v42 == 11 )
              {
LABEL_66:
                if ( v33 )
                {
                  v40 = *(_QWORD *)(v34 + 96);
                  if ( *((_DWORD *)v33 + 2) <= 0x40u && *(_DWORD *)(v40 + 32) <= 0x40u )
                  {
                    *v33 = *(_QWORD *)(v40 + 24);
                    *((_DWORD *)v33 + 2) = *(_DWORD *)(v40 + 32);
                  }
                  else
                  {
                    sub_C43990((__int64)v33, v40 + 24);
                  }
                }
                goto LABEL_54;
              }
            }
            v53 = 1;
            if ( !v33 )
              v33 = &v52;
            v52 = 0;
            v43 = sub_33D1410(v34, v33);
            if ( v53 > 0x40 && v52 )
              j_j___libc_free_0_0(v52);
            if ( v43 )
              goto LABEL_54;
            v8 = *(_QWORD **)(a1 + 40);
          }
          goto LABEL_24;
        }
        if ( v18 )
        {
          v46 = *(_QWORD *)(v19 + 96);
          if ( *((_DWORD *)v18 + 2) <= 0x40u && *(_DWORD *)(v46 + 32) <= 0x40u )
          {
            *v18 = *(_QWORD *)(v46 + 24);
            *((_DWORD *)v18 + 2) = *(_DWORD *)(v46 + 32);
          }
          else
          {
            sub_C43990(*(_QWORD *)(a4 + 24), v46 + 24);
          }
        }
      }
    }
    v22 = *(_QWORD **)(a1 + 40);
    goto LABEL_75;
  }
LABEL_4:
  v11 = v8[5];
  if ( v9 != *(_DWORD *)(v11 + 24) )
    return 0;
  v23 = *(_QWORD *)(a4 + 16);
  v49 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v11 + 40));
  *(_QWORD *)v23 = v49.m128i_i64[0];
  *(_DWORD *)(v23 + 8) = v49.m128i_i32[2];
  v24 = *(unsigned __int64 **)(a4 + 24);
  v25 = *(_QWORD *)(*(_QWORD *)(v11 + 40) + 40LL);
  if ( v25 )
  {
    v26 = *(_DWORD *)(v25 + 24);
    if ( v26 == 35 || v26 == 11 )
    {
      if ( !v24 )
        goto LABEL_43;
      v44 = *(_QWORD *)(v25 + 96);
      v45 = v44 + 24;
      if ( *((_DWORD *)v24 + 2) <= 0x40u && *(_DWORD *)(v44 + 32) <= 0x40u )
      {
        *v24 = *(_QWORD *)(v44 + 24);
        *((_DWORD *)v24 + 2) = *(_DWORD *)(v44 + 32);
        goto LABEL_43;
      }
LABEL_90:
      sub_C43990(*(_QWORD *)(a4 + 24), v45);
      goto LABEL_43;
    }
  }
  v53 = 1;
  if ( !v24 )
    v24 = &v52;
  v52 = 0;
  v27 = sub_33D1410(v25, v24);
  if ( v53 > 0x40 && v52 )
    j_j___libc_free_0_0(v52);
  if ( v27 )
    goto LABEL_43;
  v28 = *(_QWORD *)(a4 + 16);
  v48 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v11 + 40) + 40LL));
  *(_QWORD *)v28 = v48.m128i_i64[0];
  *(_DWORD *)(v28 + 8) = v48.m128i_i32[2];
  v29 = *(unsigned __int64 **)(a4 + 24);
  v30 = **(_QWORD **)(v11 + 40);
  if ( v30 )
  {
    v31 = *(_DWORD *)(v30 + 24);
    if ( v31 == 11 || v31 == 35 )
    {
      if ( !v29 )
        goto LABEL_43;
      v47 = *(_QWORD *)(v30 + 96);
      v45 = v47 + 24;
      if ( *((_DWORD *)v29 + 2) <= 0x40u && *(_DWORD *)(v47 + 32) <= 0x40u )
      {
        *v29 = *(_QWORD *)(v47 + 24);
        *((_DWORD *)v29 + 2) = *(_DWORD *)(v47 + 32);
        goto LABEL_43;
      }
      goto LABEL_90;
    }
  }
  v53 = 1;
  if ( !v29 )
    v29 = &v52;
  v52 = 0;
  v32 = sub_33D1410(v30, v29);
  if ( v53 > 0x40 && v52 )
    j_j___libc_free_0_0(v52);
  if ( !v32 )
    return 0;
LABEL_43:
  if ( *(_BYTE *)(a4 + 36) && *(_DWORD *)(a4 + 32) != (*(_DWORD *)(a4 + 32) & *(_DWORD *)(v11 + 28)) )
    return 0;
  v33 = *(unsigned __int64 **)(a4 + 40);
  v34 = **(_QWORD **)(a1 + 40);
  if ( v34 )
  {
    v35 = *(_DWORD *)(v34 + 24);
    if ( v35 == 35 || v35 == 11 )
      goto LABEL_66;
  }
  v53 = 1;
  if ( !v33 )
    v33 = &v52;
  v52 = 0;
  v36 = sub_33D1410(v34, v33);
  if ( v53 > 0x40 && v52 )
    j_j___libc_free_0_0(v52);
  if ( !v36 )
    return 0;
LABEL_54:
  if ( *(_BYTE *)(a4 + 52) && *(_DWORD *)(a4 + 48) != (*(_DWORD *)(a4 + 48) & *(_DWORD *)(a1 + 28)) )
    return 0;
  v37 = *(_QWORD *)(a1 + 56);
  if ( !v37 )
    return 0;
  v38 = 1;
  while ( 1 )
  {
    while ( *(_DWORD *)(v37 + 8) != a2 )
    {
      v37 = *(_QWORD *)(v37 + 32);
      if ( !v37 )
        return v38 ^ 1u;
    }
    if ( !v38 )
      return 0;
    v39 = *(_QWORD *)(v37 + 32);
    if ( !v39 )
      break;
    if ( a2 == *(_DWORD *)(v39 + 8) )
      return 0;
    v37 = *(_QWORD *)(v39 + 32);
    v38 = 0;
    if ( !v37 )
      return v38 ^ 1u;
  }
  return 1;
}
