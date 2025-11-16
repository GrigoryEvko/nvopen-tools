// Function: sub_12F7D90
// Address: 0x12f7d90
//
__int64 __fastcall sub_12F7D90(int a1, const void **a2, __int64 a3)
{
  __int64 result; // rax
  const char *v5; // rbx
  size_t v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rdx
  _QWORD *v9; // rdi
  size_t v10; // rax
  size_t v11; // r14
  _QWORD *v12; // rdx
  _QWORD *v13; // rdi
  size_t v14; // rax
  size_t v15; // r14
  _QWORD *v16; // rdx
  __int64 v17; // rcx
  size_t v18; // rax
  size_t v19; // r14
  _QWORD *v20; // rdx
  char v21; // si
  _QWORD *v22; // rdi
  size_t v23; // rax
  size_t v24; // r14
  _QWORD *v25; // rdx
  _QWORD *v26; // rdi
  size_t v27; // rax
  size_t v28; // r14
  _QWORD *v29; // rdx
  __int64 v30; // rax
  _QWORD *v31; // rdi
  _QWORD *v32; // rdi
  _QWORD *v33; // rdi
  __int64 v35; // [rsp+38h] [rbp-288h]
  const void **v36; // [rsp+58h] [rbp-268h]
  int v37; // [rsp+6Ch] [rbp-254h] BYREF
  _BYTE *v38; // [rsp+70h] [rbp-250h] BYREF
  size_t v39; // [rsp+78h] [rbp-248h]
  _QWORD v40[18]; // [rsp+80h] [rbp-240h] BYREF
  __int64 (__fastcall **v41)(); // [rsp+110h] [rbp-1B0h] BYREF
  __int64 v42; // [rsp+118h] [rbp-1A8h]
  __int64 (__fastcall **v43)(); // [rsp+120h] [rbp-1A0h] BYREF
  __int64 v44; // [rsp+128h] [rbp-198h]
  __int64 v45; // [rsp+130h] [rbp-190h]
  __int64 v46; // [rsp+138h] [rbp-188h]
  __int64 v47; // [rsp+140h] [rbp-180h]
  __int64 v48; // [rsp+148h] [rbp-178h]
  __int64 v49; // [rsp+150h] [rbp-170h]
  _BYTE v50[8]; // [rsp+158h] [rbp-168h] BYREF
  int v51; // [rsp+160h] [rbp-160h]
  __int64 v52[2]; // [rsp+168h] [rbp-158h] BYREF
  _QWORD v53[2]; // [rsp+178h] [rbp-148h] BYREF
  _QWORD v54[27]; // [rsp+188h] [rbp-138h] BYREF
  __int64 v55; // [rsp+260h] [rbp-60h]
  __int16 v56; // [rsp+268h] [rbp-58h]
  __int64 v57; // [rsp+270h] [rbp-50h]
  __int64 v58; // [rsp+278h] [rbp-48h]
  __int64 v59; // [rsp+280h] [rbp-40h]
  __int64 v60; // [rsp+288h] [rbp-38h]

  memset(&v40[2], 0, 0x78u);
  result = sub_1C13890(a3);
  *(_QWORD *)(a3 + 4) = 0x300000000LL;
  if ( a1 > 0 )
  {
    v36 = a2;
    v35 = (__int64)&a2[(unsigned int)(a1 - 1) + 1];
    while ( 1 )
    {
      v5 = (const char *)*v36;
      if ( !memcmp(*v36, "-arch=compute_", 0xEu) )
      {
        v38 = v40;
        v6 = strlen(v5 + 14);
        sub_12F7CE0((__int64 *)&v38, (_BYTE *)v5 + 14, (__int64)&v5[v6 + 14]);
        sub_222DF20(v54);
        v55 = 0;
        v56 = 0;
        v54[0] = off_4A06798;
        v57 = 0;
        v58 = 0;
        v59 = 0;
        v41 = (__int64 (__fastcall **)())qword_4A07108;
        v60 = 0;
        *(__int64 (__fastcall ***)())((char *)&v41 + qword_4A07108[-3]) = (__int64 (__fastcall **)())&unk_4A07130;
        v42 = 0;
        sub_222DD70((char *)&v41 + (_QWORD)*(v41 - 3), 0);
        v44 = 0;
        v45 = 0;
        v46 = 0;
        v41 = off_4A07178;
        v54[0] = off_4A071A0;
        v43 = off_4A07480;
        v47 = 0;
        v48 = 0;
        v49 = 0;
        sub_220A990(v50);
        v52[0] = (__int64)v53;
        v43 = off_4A07080;
        v51 = 0;
        sub_12F7CE0(v52, v38, (__int64)&v38[v39]);
        v51 = 8;
        sub_223FD50(&v43, v52[0], 0, 0);
        sub_222DD70(v54, &v43);
        v7 = (__int64)&v37;
        sub_222E4D0(&v41, &v37);
        v9 = (_QWORD *)v52[0];
        v41 = off_4A07178;
        *(_DWORD *)a3 = 10 * v37;
        v54[0] = off_4A071A0;
        v43 = off_4A07080;
        if ( v9 == v53 )
          goto LABEL_13;
        goto LABEL_12;
      }
      if ( !memcmp(*v36, "-opt=", 5u) )
        break;
      if ( !memcmp(*v36, "-ftz=", 5u) )
      {
        v38 = v40;
        v14 = strlen(v5 + 5);
        v41 = (__int64 (__fastcall **)())v14;
        v15 = v14;
        if ( v14 > 0xF )
        {
          v38 = (_BYTE *)sub_22409D0(&v38, &v41, 0);
          v26 = v38;
          v40[0] = v41;
        }
        else
        {
          if ( v14 == 1 )
          {
            LOBYTE(v40[0]) = v5[5];
            v16 = v40;
LABEL_33:
            v39 = v14;
            *((_BYTE *)v16 + v14) = 0;
            sub_222DF20(v54);
            v55 = 0;
            v54[0] = off_4A06798;
            v56 = 0;
            v57 = 0;
            v58 = 0;
            v41 = (__int64 (__fastcall **)())qword_4A07108;
            v59 = 0;
            v60 = 0;
            *(__int64 (__fastcall ***)())((char *)&v41 + qword_4A07108[-3]) = (__int64 (__fastcall **)())&unk_4A07130;
            v42 = 0;
            sub_222DD70((char *)&v41 + (_QWORD)*(v41 - 3), 0);
            v44 = 0;
            v45 = 0;
            v46 = 0;
            v41 = off_4A07178;
            v54[0] = off_4A071A0;
            v43 = off_4A07480;
            v47 = 0;
            v48 = 0;
            v49 = 0;
            sub_220A990(v50);
            v51 = 0;
            v43 = off_4A07080;
            v52[0] = (__int64)v53;
            sub_12F7CE0(v52, v38, (__int64)&v38[v39]);
            v51 = 8;
            sub_223FD50(&v43, v52[0], 0, 0);
            sub_222DD70(v54, &v43);
            sub_222E4D0(&v41, &v37);
            v17 = a3;
            v8 = 32 * (v37 & 1u);
            v7 = *(_BYTE *)(a3 + 200) & 0xDF;
            goto LABEL_34;
          }
          if ( !v14 )
          {
            v16 = v40;
            goto LABEL_33;
          }
          v26 = v40;
        }
        memcpy(v26, v5 + 5, v15);
        v14 = (size_t)v41;
        v16 = v38;
        goto LABEL_33;
      }
      if ( !memcmp(*v36, "-fma=", 5u) )
      {
        v38 = v40;
        v18 = strlen(v5 + 5);
        v41 = (__int64 (__fastcall **)())v18;
        v19 = v18;
        if ( v18 > 0xF )
        {
          v38 = (_BYTE *)sub_22409D0(&v38, &v41, 0);
          v31 = v38;
          v40[0] = v41;
        }
        else
        {
          if ( v18 == 1 )
          {
            LOBYTE(v40[0]) = v5[5];
            v20 = v40;
LABEL_40:
            v39 = v18;
            *((_BYTE *)v20 + v18) = 0;
            sub_222DF20(v54);
            v55 = 0;
            v56 = 0;
            v54[0] = off_4A06798;
            v57 = 0;
            v58 = 0;
            v59 = 0;
            v41 = (__int64 (__fastcall **)())qword_4A07108;
            v60 = 0;
            *(__int64 (__fastcall ***)())((char *)&v41 + qword_4A07108[-3]) = (__int64 (__fastcall **)())&unk_4A07130;
            v42 = 0;
            sub_222DD70((char *)&v41 + (_QWORD)*(v41 - 3), 0);
            v44 = 0;
            v45 = 0;
            v46 = 0;
            v41 = off_4A07178;
            v54[0] = off_4A071A0;
            v43 = off_4A07480;
            v47 = 0;
            v48 = 0;
            v49 = 0;
            sub_220A990(v50);
            v51 = 0;
            v43 = off_4A07080;
            v52[0] = (__int64)v53;
            sub_12F7CE0(v52, v38, (__int64)&v38[v39]);
            v51 = 8;
            sub_223FD50(&v43, v52[0], 0, 0);
            sub_222DD70(v54, &v43);
            sub_222E4D0(&v41, &v37);
            v21 = *(_BYTE *)(a3 + 200);
            v8 = (unsigned __int8)v37 << 7;
            v41 = off_4A07178;
            v7 = v21 & 0x7F;
            v54[0] = off_4A071A0;
            *(_BYTE *)(a3 + 200) = ((_BYTE)v37 << 7) | v7;
            goto LABEL_35;
          }
          if ( !v18 )
          {
            v20 = v40;
            goto LABEL_40;
          }
          v31 = v40;
        }
        memcpy(v31, v5 + 5, v19);
        v18 = (size_t)v41;
        v20 = v38;
        goto LABEL_40;
      }
      if ( !memcmp(*v36, "-prec-div=", 0xAu) )
      {
        v38 = v40;
        v23 = strlen(v5 + 10);
        v41 = (__int64 (__fastcall **)())v23;
        v24 = v23;
        if ( v23 > 0xF )
        {
          v38 = (_BYTE *)sub_22409D0(&v38, &v41, 0);
          v32 = v38;
          v40[0] = v41;
        }
        else
        {
          if ( v23 == 1 )
          {
            LOBYTE(v40[0]) = v5[10];
            v25 = v40;
LABEL_48:
            v39 = v23;
            *((_BYTE *)v25 + v23) = 0;
            sub_222DF20(v54);
            v56 = 0;
            v54[0] = off_4A06798;
            v55 = 0;
            v57 = 0;
            v58 = 0;
            v41 = (__int64 (__fastcall **)())qword_4A07108;
            v59 = 0;
            v60 = 0;
            *(__int64 (__fastcall ***)())((char *)&v41 + qword_4A07108[-3]) = (__int64 (__fastcall **)())&unk_4A07130;
            v42 = 0;
            sub_222DD70((char *)&v41 + (_QWORD)*(v41 - 3), 0);
            v44 = 0;
            v41 = off_4A07178;
            v54[0] = off_4A071A0;
            v43 = off_4A07480;
            v45 = 0;
            v46 = 0;
            v47 = 0;
            v48 = 0;
            v49 = 0;
            sub_220A990(v50);
            v52[0] = (__int64)v53;
            v43 = off_4A07080;
            v51 = 0;
            sub_12F7CE0(v52, v38, (__int64)&v38[v39]);
            v51 = 8;
            sub_223FD50(&v43, v52[0], 0, 0);
            sub_222DD70(v54, &v43);
            v7 = (__int64)&v37;
            sub_222E4D0(&v41, &v37);
            v43 = off_4A07080;
            v13 = (_QWORD *)v52[0];
            v41 = off_4A07178;
            *(_DWORD *)(a3 + 204) = (v37 == 0) + 1;
            v54[0] = off_4A071A0;
            if ( v13 == v53 )
              goto LABEL_13;
LABEL_26:
            v7 = v53[0] + 1LL;
            j_j___libc_free_0(v13, v53[0] + 1LL);
            goto LABEL_13;
          }
          if ( !v23 )
          {
            v25 = v40;
            goto LABEL_48;
          }
          v32 = v40;
        }
        memcpy(v32, v5 + 10, v24);
        v23 = (size_t)v41;
        v25 = v38;
        goto LABEL_48;
      }
      if ( !memcmp(*v36, "-prec-sqrt=", 0xBu) )
      {
        v38 = v40;
        v27 = strlen(v5 + 11);
        v41 = (__int64 (__fastcall **)())v27;
        v28 = v27;
        if ( v27 > 0xF )
        {
          v38 = (_BYTE *)sub_22409D0(&v38, &v41, 0);
          v33 = v38;
          v40[0] = v41;
        }
        else
        {
          if ( v27 == 1 )
          {
            LOBYTE(v40[0]) = v5[11];
            v29 = v40;
            goto LABEL_63;
          }
          if ( !v27 )
          {
            v29 = v40;
            goto LABEL_63;
          }
          v33 = v40;
        }
        memcpy(v33, v5 + 11, v28);
        v27 = (size_t)v41;
        v29 = v38;
LABEL_63:
        v39 = v27;
        *((_BYTE *)v29 + v27) = 0;
        sub_222DF20(v54);
        v55 = 0;
        v57 = 0;
        v54[0] = off_4A06798;
        v56 = 0;
        v58 = 0;
        v59 = 0;
        v60 = 0;
        v41 = (__int64 (__fastcall **)())qword_4A07108;
        *(__int64 (__fastcall ***)())((char *)&v41 + qword_4A07108[-3]) = (__int64 (__fastcall **)())&unk_4A07130;
        v42 = 0;
        sub_222DD70((char *)&v41 + (_QWORD)*(v41 - 3), 0);
        v44 = 0;
        v45 = 0;
        v46 = 0;
        v41 = off_4A07178;
        v54[0] = off_4A071A0;
        v43 = off_4A07480;
        v47 = 0;
        v48 = 0;
        v49 = 0;
        sub_220A990(v50);
        v51 = 0;
        v43 = off_4A07080;
        v52[0] = (__int64)v53;
        sub_12F7CE0(v52, v38, (__int64)&v38[v39]);
        v51 = 8;
        sub_223FD50(&v43, v52[0], 0, 0);
        sub_222DD70(v54, &v43);
        v30 = sub_222E4D0(&v41, &v37);
        v17 = a3;
        LOBYTE(v30) = v37 == 0;
        v7 = *(_BYTE *)(a3 + 200) & 0xBF;
        v8 = (unsigned int)((_DWORD)v30 << 6);
LABEL_34:
        *(_BYTE *)(v17 + 200) = v8 | v7;
        v41 = off_4A07178;
        v54[0] = off_4A071A0;
LABEL_35:
        v9 = (_QWORD *)v52[0];
        v43 = off_4A07080;
        if ( (_QWORD *)v52[0] == v53 )
        {
LABEL_13:
          v43 = off_4A07480;
          sub_2209150(v50, v7, v8);
          v41 = (__int64 (__fastcall **)())qword_4A07108;
          *(__int64 (__fastcall ***)())((char *)&v41 + qword_4A07108[-3]) = (__int64 (__fastcall **)())&unk_4A07130;
          v42 = 0;
          v54[0] = off_4A06798;
          sub_222E050(v54);
          if ( v38 != (_BYTE *)v40 )
            j_j___libc_free_0(v38, v40[0] + 1LL);
          goto LABEL_15;
        }
LABEL_12:
        v7 = v53[0] + 1LL;
        j_j___libc_free_0(v9, v53[0] + 1LL);
        goto LABEL_13;
      }
      if ( !strcmp((const char *)*v36, "--device-c") )
      {
        *(_DWORD *)(a3 + 4) = 2;
      }
      else if ( *v5 == 45 && v5[1] == 103 && !v5[2] )
      {
        *(_DWORD *)(a3 + 12) = 2;
      }
      else if ( !strcmp((const char *)*v36, "-generate-line-info") )
      {
        *(_DWORD *)(a3 + 12) = 1;
      }
LABEL_15:
      result = (__int64)++v36;
      if ( v36 == (const void **)v35 )
        return result;
    }
    v38 = v40;
    v10 = strlen(v5 + 5);
    v41 = (__int64 (__fastcall **)())v10;
    v11 = v10;
    if ( v10 > 0xF )
    {
      v38 = (_BYTE *)sub_22409D0(&v38, &v41, 0);
      v22 = v38;
      v40[0] = v41;
    }
    else
    {
      if ( v10 == 1 )
      {
        LOBYTE(v40[0]) = v5[5];
        v12 = v40;
        goto LABEL_20;
      }
      if ( !v10 )
      {
        v12 = v40;
        goto LABEL_20;
      }
      v22 = v40;
    }
    memcpy(v22, v5 + 5, v11);
    v10 = (size_t)v41;
    v12 = v38;
LABEL_20:
    v39 = v10;
    *((_BYTE *)v12 + v10) = 0;
    sub_222DF20(v54);
    v55 = 0;
    v56 = 0;
    v54[0] = off_4A06798;
    v57 = 0;
    v58 = 0;
    v59 = 0;
    v41 = (__int64 (__fastcall **)())qword_4A07108;
    v60 = 0;
    *(__int64 (__fastcall ***)())((char *)&v41 + qword_4A07108[-3]) = (__int64 (__fastcall **)())&unk_4A07130;
    v42 = 0;
    sub_222DD70((char *)&v41 + (_QWORD)*(v41 - 3), 0);
    v44 = 0;
    v45 = 0;
    v46 = 0;
    v41 = off_4A07178;
    v54[0] = off_4A071A0;
    v47 = 0;
    v43 = off_4A07480;
    v48 = 0;
    v49 = 0;
    sub_220A990(v50);
    v52[0] = (__int64)v53;
    v43 = off_4A07080;
    v51 = 0;
    sub_12F7CE0(v52, v38, (__int64)&v38[v39]);
    v51 = 8;
    sub_223FD50(&v43, v52[0], 0, 0);
    sub_222DD70(v54, &v43);
    v7 = (__int64)&v37;
    sub_222E4D0(&v41, &v37);
    if ( v37 == 2 )
    {
      *(_DWORD *)(a3 + 8) = 2;
    }
    else if ( v37 > 2 )
    {
      if ( v37 == 3 )
        *(_DWORD *)(a3 + 8) = 3;
    }
    else if ( v37 )
    {
      if ( v37 == 1 )
        *(_DWORD *)(a3 + 8) = 1;
    }
    else
    {
      *(_DWORD *)(a3 + 8) = 0;
    }
    v13 = (_QWORD *)v52[0];
    v41 = off_4A07178;
    v54[0] = off_4A071A0;
    v43 = off_4A07080;
    if ( (_QWORD *)v52[0] == v53 )
      goto LABEL_13;
    goto LABEL_26;
  }
  return result;
}
