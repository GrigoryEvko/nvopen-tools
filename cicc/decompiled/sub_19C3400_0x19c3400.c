// Function: sub_19C3400
// Address: 0x19c3400
//
_QWORD *__fastcall sub_19C3400(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v5; // r14
  _QWORD *v9; // rax
  _QWORD *v10; // r15
  _BYTE *v11; // rsi
  __int64 *v12; // rbx
  __int64 v13; // rcx
  __int64 *v15; // r14
  int v16; // edx
  __int64 v17; // rdi
  int v18; // edx
  __int64 v19; // r9
  unsigned int v20; // esi
  __int64 *v21; // rax
  __int64 v22; // r11
  unsigned int v23; // esi
  int v24; // edx
  __int64 v25; // rax
  _QWORD *v26; // r13
  int v27; // edi
  __int64 v28; // rdx
  unsigned __int64 *v29; // rdi
  __int64 v30; // rdx
  __int64 *v31; // rbx
  __int64 *i; // r14
  __int64 v33; // rdi
  __int64 v35; // r9
  unsigned int v36; // edx
  __int64 v37; // rdi
  int v38; // eax
  int v39; // r8d
  _BYTE *v40; // rsi
  int v41; // r11d
  _QWORD *v42; // r8
  int v43; // edi
  int v44; // esi
  int v45; // esi
  __int64 v46; // r9
  _QWORD *v47; // r10
  int v48; // r8d
  unsigned int v49; // edx
  __int64 v50; // rdi
  int v51; // edx
  __int64 v52; // r9
  int v53; // r8d
  unsigned int v54; // esi
  __int64 v55; // rdi
  __int64 v56; // [rsp+8h] [rbp-88h]
  __int64 v58; // [rsp+20h] [rbp-70h]
  __int64 v59; // [rsp+20h] [rbp-70h]
  __int64 v60; // [rsp+20h] [rbp-70h]
  __int64 v61; // [rsp+20h] [rbp-70h]
  __int64 v62; // [rsp+20h] [rbp-70h]
  __int64 v63; // [rsp+20h] [rbp-70h]
  __int64 v64; // [rsp+28h] [rbp-68h]
  void *v65; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v66[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v67; // [rsp+48h] [rbp-48h]
  __int64 v68; // [rsp+50h] [rbp-40h]

  v5 = a1;
  v9 = sub_194ACF0(a4);
  v65 = v9;
  v10 = v9;
  if ( a2 )
  {
    *v9 = a2;
    v11 = *(_BYTE **)(a2 + 16);
    if ( v11 == *(_BYTE **)(a2 + 24) )
    {
      sub_13FD960(a2 + 8, v11, &v65);
    }
    else
    {
      if ( v11 )
      {
        *(_QWORD *)v11 = v65;
        v11 = *(_BYTE **)(a2 + 16);
      }
      *(_QWORD *)(a2 + 16) = v11 + 8;
    }
  }
  else
  {
    v40 = *(_BYTE **)(a4 + 40);
    if ( v40 == *(_BYTE **)(a4 + 48) )
    {
      sub_13FD960(a4 + 32, v40, &v65);
    }
    else
    {
      if ( v40 )
      {
        *(_QWORD *)v40 = v9;
        v40 = *(_BYTE **)(a4 + 40);
      }
      *(_QWORD *)(a4 + 40) = v40 + 8;
    }
  }
  sub_14070E0(a5, v10);
  v12 = *(__int64 **)(a1 + 32);
  if ( *(__int64 **)(a1 + 40) != v12 )
  {
    v64 = (__int64)v10;
    v13 = a3;
    v15 = *(__int64 **)(a1 + 40);
    while ( 1 )
    {
      v16 = *(_DWORD *)(a4 + 24);
      if ( v16 )
      {
        v17 = *v12;
        v18 = v16 - 1;
        v19 = *(_QWORD *)(a4 + 8);
        v20 = v18 & (((unsigned int)*v12 >> 9) ^ ((unsigned int)*v12 >> 4));
        v21 = (__int64 *)(v19 + 16LL * v20);
        v22 = *v21;
        if ( *v12 != *v21 )
        {
          v38 = 1;
          while ( v22 != -8 )
          {
            v39 = v38 + 1;
            v20 = v18 & (v38 + v20);
            v21 = (__int64 *)(v19 + 16LL * v20);
            v22 = *v21;
            if ( v17 == *v21 )
              goto LABEL_11;
            v38 = v39;
          }
          goto LABEL_8;
        }
LABEL_11:
        if ( a1 == v21[1] )
          break;
      }
LABEL_8:
      if ( v15 == ++v12 )
      {
        v5 = a1;
        v10 = (_QWORD *)v64;
        a3 = v13;
        goto LABEL_30;
      }
    }
    v66[0] = 2;
    v66[1] = 0;
    v67 = v17;
    if ( v17 != -8 && v17 != 0 && v17 != -16 )
    {
      v58 = v13;
      sub_164C220((__int64)v66);
      v13 = v58;
    }
    v23 = *(_DWORD *)(v13 + 24);
    v68 = v13;
    v65 = &unk_49E6B50;
    if ( v23 )
    {
      v25 = v67;
      v35 = *(_QWORD *)(v13 + 8);
      v36 = (v23 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
      v26 = (_QWORD *)(v35 + ((unsigned __int64)v36 << 6));
      v37 = v26[3];
      if ( v67 == v37 )
      {
LABEL_34:
        v65 = &unk_49EE2B0;
        if ( v25 != 0 && v25 != -8 && v25 != -16 )
        {
          v61 = v13;
          sub_1649B30(v66);
          v13 = v61;
        }
        v62 = v13;
        sub_1400330(v64, v26[7], a4);
        v13 = v62;
        goto LABEL_8;
      }
      v41 = 1;
      v42 = 0;
      while ( v37 != -8 )
      {
        if ( !v42 && v37 == -16 )
          v42 = v26;
        v36 = (v23 - 1) & (v41 + v36);
        v26 = (_QWORD *)(v35 + ((unsigned __int64)v36 << 6));
        v37 = v26[3];
        if ( v67 == v37 )
          goto LABEL_34;
        ++v41;
      }
      v43 = *(_DWORD *)(v13 + 16);
      if ( v42 )
        v26 = v42;
      ++*(_QWORD *)v13;
      v27 = v43 + 1;
      if ( 4 * v27 < 3 * v23 )
      {
        if ( v23 - *(_DWORD *)(v13 + 20) - v27 > v23 >> 3 )
        {
LABEL_20:
          *(_DWORD *)(v13 + 16) = v27;
          if ( v26[3] == -8 )
          {
            v29 = v26 + 1;
            if ( v25 != -8 )
            {
LABEL_25:
              v26[3] = v25;
              if ( v25 == -8 || v25 == 0 || v25 == -16 )
              {
                v25 = v67;
              }
              else
              {
                v60 = v13;
                sub_1649AC0(v29, v66[0] & 0xFFFFFFFFFFFFFFF8LL);
                v25 = v67;
                v13 = v60;
              }
            }
          }
          else
          {
            --*(_DWORD *)(v13 + 20);
            v28 = v26[3];
            if ( v25 != v28 )
            {
              v29 = v26 + 1;
              if ( v28 != 0 && v28 != -8 && v28 != -16 )
              {
                v56 = v13;
                sub_1649B30(v29);
                v25 = v67;
                v13 = v56;
                v29 = v26 + 1;
              }
              goto LABEL_25;
            }
          }
          v30 = v68;
          v26[5] = 6;
          v26[6] = 0;
          v26[4] = v30;
          v26[7] = 0;
          goto LABEL_34;
        }
        v63 = v13;
        sub_12E48B0(v13, v23);
        v13 = v63;
        v44 = *(_DWORD *)(v63 + 24);
        if ( !v44 )
          goto LABEL_18;
        v25 = v67;
        v45 = v44 - 1;
        v46 = *(_QWORD *)(v63 + 8);
        v47 = 0;
        v48 = 1;
        v49 = v45 & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
        v26 = (_QWORD *)(v46 + ((unsigned __int64)v49 << 6));
        v50 = v26[3];
        if ( v50 == v67 )
          goto LABEL_19;
        while ( v50 != -8 )
        {
          if ( !v47 && v50 == -16 )
            v47 = v26;
          v49 = v45 & (v48 + v49);
          v26 = (_QWORD *)(v46 + ((unsigned __int64)v49 << 6));
          v50 = v26[3];
          if ( v67 == v50 )
            goto LABEL_19;
          ++v48;
        }
        goto LABEL_67;
      }
    }
    else
    {
      ++*(_QWORD *)v13;
    }
    v59 = v13;
    sub_12E48B0(v13, 2 * v23);
    v13 = v59;
    v24 = *(_DWORD *)(v59 + 24);
    if ( !v24 )
    {
LABEL_18:
      v25 = v67;
      v26 = 0;
LABEL_19:
      v27 = *(_DWORD *)(v13 + 16) + 1;
      goto LABEL_20;
    }
    v25 = v67;
    v51 = v24 - 1;
    v52 = *(_QWORD *)(v59 + 8);
    v47 = 0;
    v53 = 1;
    v54 = v51 & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
    v26 = (_QWORD *)(v52 + ((unsigned __int64)v54 << 6));
    v55 = v26[3];
    if ( v67 == v55 )
      goto LABEL_19;
    while ( v55 != -8 )
    {
      if ( v55 == -16 && !v47 )
        v47 = v26;
      v54 = v51 & (v53 + v54);
      v26 = (_QWORD *)(v52 + ((unsigned __int64)v54 << 6));
      v55 = v26[3];
      if ( v67 == v55 )
        goto LABEL_19;
      ++v53;
    }
LABEL_67:
    if ( v47 )
      v26 = v47;
    goto LABEL_19;
  }
LABEL_30:
  v31 = *(__int64 **)(v5 + 16);
  for ( i = *(__int64 **)(v5 + 8); v31 != i; ++i )
  {
    v33 = *i;
    sub_19C3400(v33, v10, a3, a4, a5);
  }
  return v10;
}
