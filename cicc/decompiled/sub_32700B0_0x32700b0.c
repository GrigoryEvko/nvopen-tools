// Function: sub_32700B0
// Address: 0x32700b0
//
__int64 __fastcall sub_32700B0(__int64 a1, __int64 a2, int a3)
{
  _QWORD *v7; // rax
  int v8; // edx
  __int64 v9; // r14
  __int64 v10; // r14
  __int64 v11; // rax
  unsigned __int64 *v12; // r8
  __int64 v13; // rdi
  int v14; // eax
  char v15; // r15
  __int64 v16; // rax
  unsigned __int64 *v17; // rax
  __int64 v18; // rdi
  int v19; // edx
  char v20; // r15
  _QWORD *v21; // rdx
  __int64 v22; // rax
  unsigned __int64 *v23; // r8
  __int64 v24; // rdi
  int v25; // eax
  char v26; // r15
  __int64 v27; // rax
  unsigned __int64 *v28; // rax
  __int64 v29; // rdi
  int v30; // edx
  char v31; // r15
  unsigned __int64 *v32; // r8
  __int64 v33; // rdi
  int v34; // eax
  char v35; // r14
  __int64 v36; // rax
  int v37; // edx
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  int v41; // eax
  char v42; // r14
  __int64 v43; // rax
  __int64 v44; // rsi
  __int64 v45; // rdx
  __int64 v46; // rdx
  __m128i v47; // [rsp-88h] [rbp-88h]
  __m128i v48; // [rsp-78h] [rbp-78h]
  __m128i v49; // [rsp-68h] [rbp-68h]
  __m128i v50; // [rsp-58h] [rbp-58h]
  unsigned __int64 v51; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v52; // [rsp-40h] [rbp-40h]

  if ( *(_DWORD *)a1 != *(_DWORD *)(a2 + 24) )
    return 0;
  v7 = *(_QWORD **)(a2 + 40);
  v8 = *(_DWORD *)(a1 + 8);
  v9 = *v7;
  if ( v8 == *(_DWORD *)(*v7 + 24LL) )
  {
    v11 = *(_QWORD *)(a1 + 16);
    v50 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v9 + 40));
    *(_QWORD *)v11 = v50.m128i_i64[0];
    *(_DWORD *)(v11 + 8) = v50.m128i_i32[2];
    v12 = *(unsigned __int64 **)(a1 + 24);
    v13 = *(_QWORD *)(*(_QWORD *)(v9 + 40) + 40LL);
    if ( v13 && ((v14 = *(_DWORD *)(v13 + 24), v14 == 35) || v14 == 11) )
    {
      if ( v12 )
      {
        v40 = *(_QWORD *)(v13 + 96);
        if ( *((_DWORD *)v12 + 2) <= 0x40u && *(_DWORD *)(v40 + 32) <= 0x40u )
        {
          *v12 = *(_QWORD *)(v40 + 24);
          *((_DWORD *)v12 + 2) = *(_DWORD *)(v40 + 32);
          v21 = *(_QWORD **)(a2 + 40);
          goto LABEL_75;
        }
        sub_C43990((__int64)v12, v40 + 24);
      }
    }
    else
    {
      v52 = 1;
      if ( !v12 )
        v12 = &v51;
      v51 = 0;
      v15 = sub_33D1410(v13, v12);
      if ( v52 > 0x40 && v51 )
        j_j___libc_free_0_0(v51);
      if ( !v15 )
      {
        v16 = *(_QWORD *)(a1 + 16);
        v49 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v9 + 40) + 40LL));
        *(_QWORD *)v16 = v49.m128i_i64[0];
        *(_DWORD *)(v16 + 8) = v49.m128i_i32[2];
        v17 = *(unsigned __int64 **)(a1 + 24);
        v18 = **(_QWORD **)(v9 + 40);
        if ( !v18 || (v19 = *(_DWORD *)(v18 + 24), v19 != 35) && v19 != 11 )
        {
          v52 = 1;
          if ( !v17 )
            v17 = &v51;
          v51 = 0;
          v20 = sub_33D1410(v18, v17);
          if ( v52 > 0x40 && v51 )
            j_j___libc_free_0_0(v51);
          v7 = *(_QWORD **)(a2 + 40);
          v21 = v7;
          if ( !v20 )
          {
LABEL_24:
            v8 = *(_DWORD *)(a1 + 8);
            goto LABEL_4;
          }
LABEL_75:
          v7 = v21;
          if ( !*(_BYTE *)(a1 + 36) || *(_DWORD *)(a1 + 32) == (*(_DWORD *)(a1 + 32) & *(_DWORD *)(v9 + 28)) )
          {
            v33 = v21[5];
            v32 = *(unsigned __int64 **)(a1 + 40);
            if ( v33 )
            {
              v41 = *(_DWORD *)(v33 + 24);
              if ( v41 == 35 || v41 == 11 )
              {
LABEL_66:
                if ( v32 )
                {
                  v39 = *(_QWORD *)(v33 + 96);
                  if ( *((_DWORD *)v32 + 2) <= 0x40u && *(_DWORD *)(v39 + 32) <= 0x40u )
                  {
                    *v32 = *(_QWORD *)(v39 + 24);
                    *((_DWORD *)v32 + 2) = *(_DWORD *)(v39 + 32);
                  }
                  else
                  {
                    sub_C43990((__int64)v32, v39 + 24);
                  }
                }
                goto LABEL_54;
              }
            }
            v52 = 1;
            if ( !v32 )
              v32 = &v51;
            v51 = 0;
            v42 = sub_33D1410(v33, v32);
            if ( v52 > 0x40 && v51 )
              j_j___libc_free_0_0(v51);
            if ( v42 )
              goto LABEL_54;
            v7 = *(_QWORD **)(a2 + 40);
          }
          goto LABEL_24;
        }
        if ( v17 )
        {
          v45 = *(_QWORD *)(v18 + 96);
          if ( *((_DWORD *)v17 + 2) <= 0x40u && *(_DWORD *)(v45 + 32) <= 0x40u )
          {
            *v17 = *(_QWORD *)(v45 + 24);
            *((_DWORD *)v17 + 2) = *(_DWORD *)(v45 + 32);
          }
          else
          {
            sub_C43990(*(_QWORD *)(a1 + 24), v45 + 24);
          }
        }
      }
    }
    v21 = *(_QWORD **)(a2 + 40);
    goto LABEL_75;
  }
LABEL_4:
  v10 = v7[5];
  if ( *(_DWORD *)(v10 + 24) != v8 )
    return 0;
  v22 = *(_QWORD *)(a1 + 16);
  v48 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v10 + 40));
  *(_QWORD *)v22 = v48.m128i_i64[0];
  *(_DWORD *)(v22 + 8) = v48.m128i_i32[2];
  v23 = *(unsigned __int64 **)(a1 + 24);
  v24 = *(_QWORD *)(*(_QWORD *)(v10 + 40) + 40LL);
  if ( v24 )
  {
    v25 = *(_DWORD *)(v24 + 24);
    if ( v25 == 35 || v25 == 11 )
    {
      if ( !v23 )
        goto LABEL_43;
      v43 = *(_QWORD *)(v24 + 96);
      v44 = v43 + 24;
      if ( *((_DWORD *)v23 + 2) <= 0x40u && *(_DWORD *)(v43 + 32) <= 0x40u )
      {
        *v23 = *(_QWORD *)(v43 + 24);
        *((_DWORD *)v23 + 2) = *(_DWORD *)(v43 + 32);
        goto LABEL_43;
      }
LABEL_90:
      sub_C43990(*(_QWORD *)(a1 + 24), v44);
      goto LABEL_43;
    }
  }
  v52 = 1;
  if ( !v23 )
    v23 = &v51;
  v51 = 0;
  v26 = sub_33D1410(v24, v23);
  if ( v52 > 0x40 && v51 )
    j_j___libc_free_0_0(v51);
  if ( v26 )
    goto LABEL_43;
  v27 = *(_QWORD *)(a1 + 16);
  v47 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v10 + 40) + 40LL));
  *(_QWORD *)v27 = v47.m128i_i64[0];
  *(_DWORD *)(v27 + 8) = v47.m128i_i32[2];
  v28 = *(unsigned __int64 **)(a1 + 24);
  v29 = **(_QWORD **)(v10 + 40);
  if ( v29 )
  {
    v30 = *(_DWORD *)(v29 + 24);
    if ( v30 == 11 || v30 == 35 )
    {
      if ( !v28 )
        goto LABEL_43;
      v46 = *(_QWORD *)(v29 + 96);
      v44 = v46 + 24;
      if ( *((_DWORD *)v28 + 2) <= 0x40u && *(_DWORD *)(v46 + 32) <= 0x40u )
      {
        *v28 = *(_QWORD *)(v46 + 24);
        *((_DWORD *)v28 + 2) = *(_DWORD *)(v46 + 32);
        goto LABEL_43;
      }
      goto LABEL_90;
    }
  }
  v52 = 1;
  if ( !v28 )
    v28 = &v51;
  v51 = 0;
  v31 = sub_33D1410(v29, v28);
  if ( v52 > 0x40 && v51 )
    j_j___libc_free_0_0(v51);
  if ( !v31 )
    return 0;
LABEL_43:
  if ( *(_BYTE *)(a1 + 36) && *(_DWORD *)(a1 + 32) != (*(_DWORD *)(a1 + 32) & *(_DWORD *)(v10 + 28)) )
    return 0;
  v32 = *(unsigned __int64 **)(a1 + 40);
  v33 = **(_QWORD **)(a2 + 40);
  if ( v33 )
  {
    v34 = *(_DWORD *)(v33 + 24);
    if ( v34 == 35 || v34 == 11 )
      goto LABEL_66;
  }
  v52 = 1;
  if ( !v32 )
    v32 = &v51;
  v51 = 0;
  v35 = sub_33D1410(v33, v32);
  if ( v52 > 0x40 && v51 )
    j_j___libc_free_0_0(v51);
  if ( !v35 )
    return 0;
LABEL_54:
  if ( *(_BYTE *)(a1 + 52) && *(_DWORD *)(a1 + 48) != (*(_DWORD *)(a1 + 48) & *(_DWORD *)(a2 + 28)) )
    return 0;
  v36 = *(_QWORD *)(a2 + 56);
  if ( !v36 )
    return 0;
  v37 = 1;
  while ( 1 )
  {
    while ( *(_DWORD *)(v36 + 8) != a3 )
    {
      v36 = *(_QWORD *)(v36 + 32);
      if ( !v36 )
        return v37 ^ 1u;
    }
    if ( !v37 )
      return 0;
    v38 = *(_QWORD *)(v36 + 32);
    if ( !v38 )
      break;
    if ( a3 == *(_DWORD *)(v38 + 8) )
      return 0;
    v36 = *(_QWORD *)(v38 + 32);
    v37 = 0;
    if ( !v36 )
      return v37 ^ 1u;
  }
  return 1;
}
