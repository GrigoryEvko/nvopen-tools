// Function: sub_194CE70
// Address: 0x194ce70
//
_QWORD *__fastcall sub_194CE70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  __int64 v6; // r13
  _QWORD *v9; // rax
  __int64 v10; // r11
  _QWORD *v11; // r15
  _BYTE *v12; // rsi
  __int64 v13; // r11
  __int64 *v14; // r10
  __int64 *v15; // rbx
  __int64 v16; // r8
  __int64 *v18; // r14
  __int64 v19; // r12
  int v20; // edx
  __int64 v21; // rdi
  int v22; // edx
  __int64 v23; // r10
  unsigned int v24; // esi
  __int64 *v25; // rax
  __int64 v26; // r9
  unsigned int v27; // esi
  int v28; // edx
  __int64 v29; // rax
  _QWORD *v30; // r13
  int v31; // edi
  __int64 v32; // rdx
  unsigned __int64 *v33; // rdi
  __int64 v34; // rdx
  __int64 *v35; // rbx
  __int64 *i; // r12
  __int64 v37; // rsi
  __int64 v39; // r9
  unsigned int v40; // edx
  __int64 v41; // rdi
  int v42; // eax
  int v43; // ecx
  __int64 v44; // rax
  _BYTE *v45; // rsi
  _QWORD *v46; // rcx
  int v47; // ecx
  int v48; // esi
  int v49; // esi
  __int64 v50; // r9
  _QWORD *v51; // rcx
  int v52; // r10d
  unsigned int v53; // edx
  __int64 v54; // rdi
  int v55; // edx
  __int64 v56; // r9
  int v57; // r10d
  unsigned int v58; // esi
  __int64 v59; // rdi
  __int64 v60; // [rsp+8h] [rbp-88h]
  __int64 v61; // [rsp+18h] [rbp-78h]
  __int64 v62; // [rsp+18h] [rbp-78h]
  __int64 v63; // [rsp+18h] [rbp-78h]
  __int64 v64; // [rsp+18h] [rbp-78h]
  __int64 v65; // [rsp+18h] [rbp-78h]
  __int64 v66; // [rsp+18h] [rbp-78h]
  __int64 v67; // [rsp+18h] [rbp-78h]
  __int64 v68; // [rsp+20h] [rbp-70h]
  __int64 v69; // [rsp+20h] [rbp-70h]
  __int64 v70; // [rsp+20h] [rbp-70h]
  __int64 v71; // [rsp+20h] [rbp-70h]
  __int64 v72; // [rsp+20h] [rbp-70h]
  int v73; // [rsp+20h] [rbp-70h]
  __int64 v74; // [rsp+20h] [rbp-70h]
  __int64 v76; // [rsp+28h] [rbp-68h]
  __int64 v77; // [rsp+28h] [rbp-68h]
  __int64 v78; // [rsp+28h] [rbp-68h]
  void *v79; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v80[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v81; // [rsp+48h] [rbp-48h]
  __int64 v82; // [rsp+50h] [rbp-40h]

  v6 = a1;
  v9 = sub_194ACF0(*(_QWORD *)(a1 + 32));
  v10 = a2;
  v11 = v9;
  if ( a3 )
  {
    v79 = v9;
    *v9 = a3;
    v12 = *(_BYTE **)(a3 + 16);
    if ( v12 == *(_BYTE **)(a3 + 24) )
    {
      v78 = v10;
      sub_13FD960(a3 + 8, v12, &v79);
      v10 = v78;
    }
    else
    {
      if ( v12 )
      {
        *(_QWORD *)v12 = v79;
        v12 = *(_BYTE **)(a3 + 16);
      }
      *(_QWORD *)(a3 + 16) = v12 + 8;
    }
  }
  else
  {
    v44 = *(_QWORD *)(a1 + 32);
    v79 = v11;
    v45 = *(_BYTE **)(v44 + 40);
    if ( v45 == *(_BYTE **)(v44 + 48) )
    {
      sub_13FD960(v44 + 32, v45, &v79);
      v10 = a2;
    }
    else
    {
      if ( v45 )
      {
        *(_QWORD *)v45 = v11;
        v45 = *(_BYTE **)(v44 + 40);
      }
      *(_QWORD *)(v44 + 40) = v45 + 8;
    }
  }
  v76 = v10;
  (*(void (__fastcall **)(_QWORD, _QWORD *, _QWORD))(a1 + 40))(*(_QWORD *)(a1 + 48), v11, a5);
  v13 = v76;
  v14 = *(__int64 **)(v76 + 40);
  v15 = *(__int64 **)(v76 + 32);
  if ( v15 != v14 )
  {
    v77 = (__int64)v11;
    v16 = a4;
    v18 = v14;
    while ( 1 )
    {
      v19 = *(_QWORD *)(a1 + 32);
      v20 = *(_DWORD *)(v19 + 24);
      if ( v20 )
      {
        v21 = *v15;
        v22 = v20 - 1;
        v23 = *(_QWORD *)(v19 + 8);
        v24 = v22 & (((unsigned int)*v15 >> 9) ^ ((unsigned int)*v15 >> 4));
        v25 = (__int64 *)(v23 + 16LL * v24);
        v26 = *v25;
        if ( *v15 != *v25 )
        {
          v42 = 1;
          while ( v26 != -8 )
          {
            v43 = v42 + 1;
            v24 = v22 & (v42 + v24);
            v25 = (__int64 *)(v23 + 16LL * v24);
            v26 = *v25;
            if ( v21 == *v25 )
              goto LABEL_11;
            v42 = v43;
          }
          goto LABEL_8;
        }
LABEL_11:
        if ( v13 == v25[1] )
          break;
      }
LABEL_8:
      if ( v18 == ++v15 )
      {
        v6 = a1;
        v11 = (_QWORD *)v77;
        a4 = v16;
        goto LABEL_30;
      }
    }
    v80[0] = 2;
    v80[1] = 0;
    v81 = v21;
    if ( v21 != -8 && v21 != 0 && v21 != -16 )
    {
      v61 = v16;
      v68 = v13;
      sub_164C220((__int64)v80);
      v16 = v61;
      v13 = v68;
    }
    v27 = *(_DWORD *)(v16 + 24);
    v82 = v16;
    v79 = &unk_49E6B50;
    if ( v27 )
    {
      v29 = v81;
      v39 = *(_QWORD *)(v16 + 8);
      v40 = (v27 - 1) & (((unsigned int)v81 >> 9) ^ ((unsigned int)v81 >> 4));
      v30 = (_QWORD *)(v39 + ((unsigned __int64)v40 << 6));
      v41 = v30[3];
      if ( v81 == v41 )
      {
LABEL_34:
        v79 = &unk_49EE2B0;
        if ( v29 != 0 && v29 != -8 && v29 != -16 )
        {
          v65 = v16;
          v71 = v13;
          sub_1649B30(v80);
          v16 = v65;
          v13 = v71;
        }
        v66 = v16;
        v72 = v13;
        sub_1400330(v77, v30[7], v19);
        v13 = v72;
        v16 = v66;
        goto LABEL_8;
      }
      v73 = 1;
      v46 = 0;
      while ( v41 != -8 )
      {
        if ( !v46 && v41 == -16 )
          v46 = v30;
        v40 = (v27 - 1) & (v73 + v40);
        v30 = (_QWORD *)(v39 + ((unsigned __int64)v40 << 6));
        v41 = v30[3];
        if ( v81 == v41 )
          goto LABEL_34;
        ++v73;
      }
      if ( v46 )
        v30 = v46;
      v47 = *(_DWORD *)(v16 + 16);
      ++*(_QWORD *)v16;
      v31 = v47 + 1;
      if ( 4 * (v47 + 1) < 3 * v27 )
      {
        if ( v27 - *(_DWORD *)(v16 + 20) - v31 > v27 >> 3 )
        {
LABEL_20:
          *(_DWORD *)(v16 + 16) = v31;
          if ( v30[3] == -8 )
          {
            v33 = v30 + 1;
            if ( v29 != -8 )
            {
LABEL_25:
              v30[3] = v29;
              if ( v29 == -8 || v29 == 0 || v29 == -16 )
              {
                v29 = v81;
              }
              else
              {
                v64 = v16;
                v70 = v13;
                sub_1649AC0(v33, v80[0] & 0xFFFFFFFFFFFFFFF8LL);
                v29 = v81;
                v16 = v64;
                v13 = v70;
              }
            }
          }
          else
          {
            --*(_DWORD *)(v16 + 20);
            v32 = v30[3];
            if ( v32 != v29 )
            {
              v33 = v30 + 1;
              if ( v32 != 0 && v32 != -8 && v32 != -16 )
              {
                v60 = v16;
                v63 = v13;
                sub_1649B30(v33);
                v29 = v81;
                v16 = v60;
                v13 = v63;
                v33 = v30 + 1;
              }
              goto LABEL_25;
            }
          }
          v34 = v82;
          v30[5] = 6;
          v30[6] = 0;
          v30[4] = v34;
          v30[7] = 0;
          goto LABEL_34;
        }
        v67 = v13;
        v74 = v16;
        sub_12E48B0(v16, v27);
        v16 = v74;
        v13 = v67;
        v48 = *(_DWORD *)(v74 + 24);
        if ( !v48 )
          goto LABEL_18;
        v29 = v81;
        v49 = v48 - 1;
        v50 = *(_QWORD *)(v74 + 8);
        v51 = 0;
        v52 = 1;
        v53 = v49 & (((unsigned int)v81 >> 9) ^ ((unsigned int)v81 >> 4));
        v30 = (_QWORD *)(v50 + ((unsigned __int64)v53 << 6));
        v54 = v30[3];
        if ( v54 == v81 )
          goto LABEL_19;
        while ( v54 != -8 )
        {
          if ( !v51 && v54 == -16 )
            v51 = v30;
          v53 = v49 & (v52 + v53);
          v30 = (_QWORD *)(v50 + ((unsigned __int64)v53 << 6));
          v54 = v30[3];
          if ( v81 == v54 )
            goto LABEL_19;
          ++v52;
        }
        goto LABEL_67;
      }
    }
    else
    {
      ++*(_QWORD *)v16;
    }
    v62 = v13;
    v69 = v16;
    sub_12E48B0(v16, 2 * v27);
    v16 = v69;
    v13 = v62;
    v28 = *(_DWORD *)(v69 + 24);
    if ( !v28 )
    {
LABEL_18:
      v29 = v81;
      v30 = 0;
LABEL_19:
      v31 = *(_DWORD *)(v16 + 16) + 1;
      goto LABEL_20;
    }
    v29 = v81;
    v55 = v28 - 1;
    v56 = *(_QWORD *)(v69 + 8);
    v51 = 0;
    v57 = 1;
    v58 = v55 & (((unsigned int)v81 >> 9) ^ ((unsigned int)v81 >> 4));
    v30 = (_QWORD *)(v56 + ((unsigned __int64)v58 << 6));
    v59 = v30[3];
    if ( v81 == v59 )
      goto LABEL_19;
    while ( v59 != -8 )
    {
      if ( !v51 && v59 == -16 )
        v51 = v30;
      v58 = v55 & (v57 + v58);
      v30 = (_QWORD *)(v56 + ((unsigned __int64)v58 << 6));
      v59 = v30[3];
      if ( v81 == v59 )
        goto LABEL_19;
      ++v57;
    }
LABEL_67:
    if ( v51 )
      v30 = v51;
    goto LABEL_19;
  }
LABEL_30:
  v35 = *(__int64 **)(v13 + 16);
  for ( i = *(__int64 **)(v13 + 8); v35 != i; ++i )
  {
    v37 = *i;
    sub_194CE70(v6, v37, v11, a4, 1);
  }
  return v11;
}
