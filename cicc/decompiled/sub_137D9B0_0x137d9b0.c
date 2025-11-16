// Function: sub_137D9B0
// Address: 0x137d9b0
//
void __fastcall sub_137D9B0(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rax
  _QWORD *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rdi
  int v10; // eax
  __int64 v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // r15
  __int64 *v14; // rax
  __int64 *v15; // rax
  char v16; // dl
  char v17; // r9
  unsigned __int64 v18; // rdx
  __int64 *v19; // r14
  __int64 v20; // rax
  _QWORD *v21; // rax
  __int64 *v22; // rcx
  __int64 *v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  __m128i *v27; // rax
  __int64 v28; // rdx
  __int64 *v29; // rdx
  __int64 *v30; // rsi
  __int64 *v31; // rcx
  __int64 v32; // rbx
  _QWORD *v33; // rax
  _QWORD *v34; // rdx
  __int64 *v35; // rdx
  __m128i v36; // [rsp+10h] [rbp-200h] BYREF
  __int64 v37; // [rsp+20h] [rbp-1F0h]
  __int64 v38; // [rsp+30h] [rbp-1E0h] BYREF
  __int64 *v39; // [rsp+38h] [rbp-1D8h]
  __int64 *v40; // [rsp+40h] [rbp-1D0h]
  __int64 v41; // [rsp+48h] [rbp-1C8h]
  int v42; // [rsp+50h] [rbp-1C0h]
  _QWORD v43[9]; // [rsp+58h] [rbp-1B8h] BYREF
  __int64 v44; // [rsp+A0h] [rbp-170h] BYREF
  _QWORD *v45; // [rsp+A8h] [rbp-168h]
  _QWORD *v46; // [rsp+B0h] [rbp-160h]
  __int64 v47; // [rsp+B8h] [rbp-158h]
  int v48; // [rsp+C0h] [rbp-150h]
  _QWORD v49[9]; // [rsp+C8h] [rbp-148h] BYREF
  _QWORD *v50; // [rsp+110h] [rbp-100h] BYREF
  __int64 v51; // [rsp+118h] [rbp-F8h]
  _QWORD v52[2]; // [rsp+120h] [rbp-F0h] BYREF
  int v53; // [rsp+130h] [rbp-E0h]

  v3 = *(_QWORD *)(a1 + 80);
  if ( v3 )
    v3 -= 24;
  v4 = sub_157EBA0(v3);
  if ( !v4 || !(unsigned int)sub_15F4D60(v4) )
    return;
  v44 = 0;
  v39 = v43;
  v40 = v43;
  v51 = 0x800000000LL;
  v45 = v49;
  v46 = v49;
  v50 = v52;
  v47 = 8;
  v48 = 0;
  v41 = 0x100000008LL;
  v42 = 0;
  v43[0] = v3;
  v38 = 1;
  v5 = sub_157EBA0(v3);
  v52[0] = v3;
  v6 = v52;
  v53 = 0;
  LODWORD(v51) = 1;
  HIDWORD(v47) = 1;
  v49[0] = v3;
  v44 = 1;
  v52[1] = v5;
  v7 = 1;
  while ( 2 )
  {
    v8 = (__int64)&v6[3 * v7 - 3];
    while ( 1 )
    {
      v9 = sub_157EBA0(v3);
      v10 = 0;
      if ( v9 )
        v10 = sub_15F4D60(v9);
      v11 = *(unsigned int *)(v8 + 16);
      if ( (_DWORD)v11 == v10 )
      {
        v32 = v50[3 * (unsigned int)v51 - 3];
        LODWORD(v51) = v51 - 1;
        v33 = v45;
        if ( v46 == v45 )
        {
          v34 = &v45[HIDWORD(v47)];
          if ( v45 == v34 )
          {
LABEL_78:
            v33 = &v45[HIDWORD(v47)];
          }
          else
          {
            while ( v32 != *v33 )
            {
              if ( v34 == ++v33 )
                goto LABEL_78;
            }
          }
        }
        else
        {
          v33 = (_QWORD *)sub_16CC9F0(&v44, v32);
          if ( v32 == *v33 )
          {
            if ( v46 == v45 )
              v34 = &v46[HIDWORD(v47)];
            else
              v34 = &v46[(unsigned int)v47];
          }
          else
          {
            if ( v46 != v45 )
            {
LABEL_69:
              v7 = (unsigned int)v51;
              goto LABEL_39;
            }
            v33 = &v46[HIDWORD(v47)];
            v34 = v33;
          }
        }
        if ( v34 != v33 )
        {
          *v33 = -2;
          ++v48;
        }
        goto LABEL_69;
      }
      v12 = *(_QWORD *)(v8 + 8);
      *(_DWORD *)(v8 + 16) = v11 + 1;
      v13 = sub_15F4DF0(v12, v11);
      v14 = v39;
      if ( v40 == v39 )
      {
        v22 = &v39[HIDWORD(v41)];
        if ( v39 != v22 )
        {
          v23 = 0;
          while ( 2 )
          {
            v24 = *v14;
            if ( v13 == *v14 )
            {
LABEL_43:
              v18 = (unsigned __int64)v46;
              v15 = v45;
              if ( v46 == v45 )
                goto LABEL_44;
              goto LABEL_13;
            }
            while ( v24 == -2 )
            {
              v29 = v14 + 1;
              v23 = v14;
              if ( v22 == v14 + 1 )
                goto LABEL_33;
              ++v14;
              v24 = *v29;
              if ( v13 == v24 )
                goto LABEL_43;
            }
            if ( v22 != ++v14 )
              continue;
            break;
          }
          if ( v23 )
          {
LABEL_33:
            *v23 = v13;
            v18 = (unsigned __int64)v46;
            --v42;
            v15 = v45;
            ++v38;
            goto LABEL_34;
          }
        }
        if ( HIDWORD(v41) < (unsigned int)v41 )
          break;
      }
      sub_16CCBA0(&v38, v13);
      v15 = v45;
      v17 = v16;
      v18 = (unsigned __int64)v46;
      if ( v17 )
      {
LABEL_34:
        if ( v15 == (__int64 *)v18 )
          goto LABEL_58;
        goto LABEL_35;
      }
      if ( v46 == v45 )
      {
LABEL_44:
        v19 = &v15[HIDWORD(v47)];
        if ( v19 == v15 )
        {
          v35 = v15;
        }
        else
        {
          do
          {
            if ( v13 == *v15 )
              break;
            ++v15;
          }
          while ( v19 != v15 );
          v35 = v19;
        }
        goto LABEL_51;
      }
LABEL_13:
      v19 = (__int64 *)(v18 + 8LL * (unsigned int)v47);
      v15 = (__int64 *)sub_16CC9F0(&v44, v13);
      if ( v13 == *v15 )
      {
        if ( v46 == v45 )
          v35 = &v46[HIDWORD(v47)];
        else
          v35 = &v46[(unsigned int)v47];
LABEL_51:
        while ( v35 != v15 && (unsigned __int64)*v15 >= 0xFFFFFFFFFFFFFFFELL )
          ++v15;
        goto LABEL_16;
      }
      if ( v46 == v45 )
      {
        v15 = &v46[HIDWORD(v47)];
        v35 = v15;
        goto LABEL_51;
      }
      v15 = &v46[(unsigned int)v47];
LABEL_16:
      if ( v15 != v19 )
      {
        v20 = *(unsigned int *)(a2 + 8);
        if ( (unsigned int)v20 >= *(_DWORD *)(a2 + 12) )
        {
          sub_16CD150(a2, a2 + 16, 0, 16);
          v20 = *(unsigned int *)(a2 + 8);
        }
        v21 = (_QWORD *)(*(_QWORD *)a2 + 16 * v20);
        *v21 = v3;
        v21[1] = v13;
        ++*(_DWORD *)(a2 + 8);
      }
    }
    ++HIDWORD(v41);
    *v22 = v13;
    v15 = v45;
    ++v38;
    if ( v45 == v46 )
    {
LABEL_58:
      v30 = &v15[HIDWORD(v47)];
      if ( v15 != v30 )
      {
        v31 = 0;
        while ( v13 != *v15 )
        {
          if ( *v15 == -2 )
            v31 = v15;
          if ( v30 == ++v15 )
          {
            if ( !v31 )
              goto LABEL_80;
            *v31 = v13;
            --v48;
            ++v44;
            goto LABEL_36;
          }
        }
        goto LABEL_36;
      }
LABEL_80:
      if ( HIDWORD(v47) < (unsigned int)v47 )
      {
        ++HIDWORD(v47);
        *v30 = v13;
        ++v44;
        goto LABEL_36;
      }
    }
LABEL_35:
    sub_16CCBA0(&v44, v13);
LABEL_36:
    v25 = sub_157EBA0(v13);
    v36.m128i_i64[0] = v13;
    v36.m128i_i64[1] = v25;
    v26 = (unsigned int)v51;
    LODWORD(v37) = 0;
    if ( (unsigned int)v51 >= HIDWORD(v51) )
    {
      sub_16CD150(&v50, v52, 0, 24);
      v26 = (unsigned int)v51;
    }
    v27 = (__m128i *)&v50[3 * v26];
    v28 = v37;
    *v27 = _mm_load_si128(&v36);
    v27[1].m128i_i64[0] = v28;
    v7 = (unsigned int)(v51 + 1);
    LODWORD(v51) = v51 + 1;
LABEL_39:
    if ( (_DWORD)v7 )
    {
      v6 = v50;
      v3 = v50[3 * v7 - 3];
      continue;
    }
    break;
  }
  if ( v46 != v45 )
    _libc_free((unsigned __int64)v46);
  if ( v50 != v52 )
    _libc_free((unsigned __int64)v50);
  if ( v40 != v39 )
    _libc_free((unsigned __int64)v40);
}
