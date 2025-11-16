// Function: sub_13960C0
// Address: 0x13960c0
//
__int64 __fastcall sub_13960C0(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v5; // rbx
  __int64 v6; // rsi
  char v7; // al
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r12
  int v11; // r12d
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r12
  int v18; // r12d
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 *v21; // rbx
  __int64 v22; // r12
  __int64 v23; // r13
  __int64 *v26; // rax
  __int64 *v27; // rdx
  __int64 v28; // rcx
  __int32 v29; // r8d
  int v30; // r9d
  __int64 *v31; // rbx
  __int64 v32; // rax
  __int64 v33; // r13
  __int64 v34; // r10
  __int64 v35; // rax
  int v36; // r11d
  unsigned int v37; // ecx
  __int64 *v38; // rdx
  __int64 v39; // rdi
  __int64 *v40; // rax
  __int64 v41; // rcx
  __int64 v42; // rdi
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rsi
  __int64 v46; // rsi
  _QWORD *v47; // r13
  _QWORD *v48; // rbx
  int v49; // edx
  int v50; // [rsp+Ch] [rbp-B4h]
  __int64 v51; // [rsp+10h] [rbp-B0h]
  __int64 v52; // [rsp+18h] [rbp-A8h]
  __int64 v53; // [rsp+20h] [rbp-A0h]
  __int32 v54; // [rsp+20h] [rbp-A0h]
  int v55; // [rsp+20h] [rbp-A0h]
  __int64 *v56; // [rsp+28h] [rbp-98h]
  __int64 v57; // [rsp+30h] [rbp-90h]
  __int64 *v58; // [rsp+30h] [rbp-90h]
  __int64 v59; // [rsp+38h] [rbp-88h]
  __m128i v60; // [rsp+40h] [rbp-80h] BYREF
  __int64 v61; // [rsp+50h] [rbp-70h]
  __int64 v62; // [rsp+60h] [rbp-60h] BYREF
  __int64 v63; // [rsp+68h] [rbp-58h]
  __int64 v64; // [rsp+70h] [rbp-50h]
  __int64 v65; // [rsp+78h] [rbp-48h]
  char v66; // [rsp+88h] [rbp-38h]

  v5 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v6 = a2 & 4;
  v7 = *(_BYTE *)(v5 + 23);
  if ( !(_DWORD)v6 )
  {
    if ( v7 < 0 )
    {
      v15 = sub_1648A40(v5);
      v17 = v15 + v16;
      if ( *(char *)(v5 + 23) >= 0 )
      {
        if ( (unsigned int)(v17 >> 4) )
          goto LABEL_72;
      }
      else if ( (unsigned int)((v17 - sub_1648A40(v5)) >> 4) )
      {
        if ( *(char *)(v5 + 23) >= 0 )
          goto LABEL_72;
        v18 = *(_DWORD *)(sub_1648A40(v5) + 8);
        if ( *(char *)(v5 + 23) >= 0 )
          goto LABEL_73;
        v19 = sub_1648A40(v5);
        v14 = -72 - 24LL * (unsigned int)(*(_DWORD *)(v19 + v20 - 4) - v18);
        goto LABEL_16;
      }
    }
    v14 = -72;
    goto LABEL_16;
  }
  if ( v7 < 0 )
  {
    v8 = sub_1648A40(v5);
    v10 = v8 + v9;
    if ( *(char *)(v5 + 23) >= 0 )
    {
      if ( (unsigned int)(v10 >> 4) )
        goto LABEL_72;
    }
    else if ( (unsigned int)((v10 - sub_1648A40(v5)) >> 4) )
    {
      if ( *(char *)(v5 + 23) < 0 )
      {
        v11 = *(_DWORD *)(sub_1648A40(v5) + 8);
        if ( *(char *)(v5 + 23) < 0 )
        {
          v12 = sub_1648A40(v5);
          v14 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v12 + v13 - 4) - v11);
          goto LABEL_16;
        }
LABEL_73:
        BUG();
      }
LABEL_72:
      BUG();
    }
  }
  v14 = -24;
LABEL_16:
  if ( -1431655765 * (unsigned int)((v14 - -24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF)) >> 3) > 0x32 )
    return 0;
  v21 = *(__int64 **)a3;
  v22 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
  if ( *(_QWORD *)a3 != v22 )
  {
    v57 = a3;
    do
    {
      v23 = *v21;
      if ( (unsigned __int8)sub_15E4F60(*v21) )
        return 0;
      if ( (unsigned __int8)sub_15E4B50(v23, v6) )
        return 0;
      if ( *(_DWORD *)(*(_QWORD *)(v23 + 24) + 8LL) >> 8 )
        return 0;
      v6 = v23;
      if ( !sub_1396090(*a1, v23) )
        return 0;
      ++v21;
    }
    while ( (__int64 *)v22 != v21 );
    v51 = *(_QWORD *)v57 + 8LL * *(unsigned int *)(v57 + 8);
    if ( *(_QWORD *)v57 != v51 )
    {
      v58 = *(__int64 **)v57;
      while ( 1 )
      {
        v26 = sub_1396090(*a1, *v58);
        v31 = (__int64 *)*v26;
        v56 = v26;
        v59 = *v26 + 24LL * *((unsigned int *)v26 + 2);
        if ( v59 != *v26 )
          break;
LABEL_48:
        v47 = (_QWORD *)v56[26];
        v48 = &v47[2 * *((unsigned int *)v56 + 54)];
        while ( v48 != v47 )
        {
          while ( 1 )
          {
            sub_14C8310(&v62, *v47, v47[1], a2);
            if ( (_BYTE)v65 )
              break;
            v47 += 2;
            if ( v48 == v47 )
              goto LABEL_53;
          }
          v47 += 2;
          sub_13848E0(a1[3], v62, v63, v64);
        }
LABEL_53:
        if ( (__int64 *)v51 == ++v58 )
          return 1;
      }
      while ( 1 )
      {
        while ( 1 )
        {
          sub_14C8270((unsigned int)&v62, a2, (_DWORD)v27, v28, v29, v30, *v31, v31[1], v31[2]);
          if ( v66 )
            break;
LABEL_29:
          v31 += 3;
          if ( (__int64 *)v59 == v31 )
            goto LABEL_48;
        }
        sub_13848E0(a1[3], v62, v63, 0);
        sub_13848E0(a1[3], v64, v65, 0);
        v32 = a1[3];
        v33 = v62;
        v29 = v63;
        v34 = *(_QWORD *)(v32 + 8);
        v35 = *(unsigned int *)(v32 + 24);
        v30 = v65;
        if ( !(_DWORD)v35 )
        {
          v42 = 0;
          goto LABEL_58;
        }
        v36 = v35 - 1;
        v37 = (v35 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
        v38 = (__int64 *)(v34 + 32LL * v37);
        v39 = *v38;
        if ( v62 == *v38 )
        {
LABEL_33:
          v40 = (__int64 *)(v34 + 32 * v35);
          if ( v40 != v38 )
          {
            v41 = v38[1];
            if ( (unsigned int)v63 < -1227133513 * (unsigned int)((v38[2] - v41) >> 3) )
            {
              v42 = v41 + 56LL * (unsigned int)v63;
              goto LABEL_36;
            }
          }
        }
        else
        {
          v49 = 1;
          while ( v39 != -8 )
          {
            v37 = v36 & (v49 + v37);
            v55 = v49 + 1;
            v38 = (__int64 *)(v34 + 32LL * v37);
            v39 = *v38;
            if ( v62 == *v38 )
              goto LABEL_33;
            v49 = v55;
          }
          v40 = (__int64 *)(v34 + 32 * v35);
        }
        v42 = 0;
LABEL_36:
        v27 = (__int64 *)(v34 + 32LL * (v36 & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4))));
        v28 = *v27;
        v53 = *v27;
        if ( *v27 == v64 )
        {
LABEL_37:
          if ( v27 != v40 )
          {
            v28 = v27[1];
            v43 = v27[2];
            LODWORD(v27) = -1227133513;
            if ( (unsigned int)v65 < -1227133513 * (unsigned int)((v43 - v28) >> 3) )
            {
              LODWORD(v27) = v65;
              v44 = v28 + 56LL * (unsigned int)v65;
              goto LABEL_40;
            }
          }
        }
        else
        {
          LODWORD(v28) = v36 & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
          LODWORD(v27) = 1;
          while ( v53 != -8 )
          {
            LODWORD(v28) = v36 & ((_DWORD)v27 + v28);
            v50 = (_DWORD)v27 + 1;
            v27 = (__int64 *)(v34 + 32LL * (unsigned int)v28);
            v53 = *v27;
            if ( *v27 == v64 )
              goto LABEL_37;
            LODWORD(v27) = v50;
          }
        }
LABEL_58:
        v44 = 0;
LABEL_40:
        v60.m128i_i64[0] = v64;
        v60.m128i_i32[2] = v65;
        v61 = 0;
        v45 = *(_QWORD *)(v42 + 8);
        if ( v45 == *(_QWORD *)(v42 + 16) )
        {
          v52 = v44;
          v54 = v63;
          sub_1384280(v42, (_BYTE *)v45, &v60);
          v44 = v52;
          v29 = v54;
        }
        else
        {
          if ( v45 )
          {
            *(__m128i *)v45 = _mm_loadu_si128(&v60);
            LODWORD(v27) = v61;
            *(_QWORD *)(v45 + 16) = v61;
            v45 = *(_QWORD *)(v42 + 8);
          }
          *(_QWORD *)(v42 + 8) = v45 + 24;
        }
        v60.m128i_i64[0] = v33;
        v60.m128i_i32[2] = v29;
        v61 = 0;
        v46 = *(_QWORD *)(v44 + 32);
        if ( v46 == *(_QWORD *)(v44 + 40) )
        {
          sub_1384280(v44 + 24, (_BYTE *)v46, &v60);
          goto LABEL_29;
        }
        if ( v46 )
        {
          *(__m128i *)v46 = _mm_loadu_si128(&v60);
          LODWORD(v27) = v61;
          *(_QWORD *)(v46 + 16) = v61;
          v46 = *(_QWORD *)(v44 + 32);
        }
        v31 += 3;
        *(_QWORD *)(v44 + 32) = v46 + 24;
        if ( (__int64 *)v59 == v31 )
          goto LABEL_48;
      }
    }
  }
  return 1;
}
