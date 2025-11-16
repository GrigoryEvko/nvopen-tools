// Function: sub_309D870
// Address: 0x309d870
//
__int64 __fastcall sub_309D870(int a1, const void **a2, __int64 a3)
{
  __int64 result; // rax
  const char *v5; // rbx
  size_t v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  _QWORD *v9; // rdi
  __int64 (__fastcall **v10)(); // rax
  size_t v11; // r14
  _QWORD *v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rcx
  _QWORD *v15; // rdi
  __int64 (__fastcall **v16)(); // rax
  size_t v17; // r14
  _QWORD *v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rcx
  char v22; // dl
  char v23; // si
  __int64 (__fastcall **v24)(); // rax
  size_t v25; // r14
  _QWORD *v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rcx
  char v29; // si
  _QWORD *v30; // rdi
  __int64 (__fastcall **v31)(); // rax
  size_t v32; // r14
  _QWORD *v33; // rdx
  __int64 v34; // rdx
  __int64 v35; // rcx
  _QWORD *v36; // rdi
  __int64 (__fastcall **v37)(); // rax
  size_t v38; // r14
  _QWORD *v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rcx
  _QWORD *v42; // rdi
  _QWORD *v43; // rdi
  _QWORD *v44; // rdi
  __int64 v46; // [rsp+38h] [rbp-288h]
  const void **v47; // [rsp+58h] [rbp-268h]
  int v48; // [rsp+6Ch] [rbp-254h] BYREF
  _BYTE *v49; // [rsp+70h] [rbp-250h] BYREF
  __int64 (__fastcall **v50)(); // [rsp+78h] [rbp-248h]
  _QWORD v51[2]; // [rsp+80h] [rbp-240h] BYREF
  __m128i v52[8]; // [rsp+90h] [rbp-230h] BYREF
  __int64 (__fastcall **v53)(); // [rsp+110h] [rbp-1B0h] BYREF
  __int64 v54; // [rsp+118h] [rbp-1A8h]
  __int64 (__fastcall **v55)(); // [rsp+120h] [rbp-1A0h] BYREF
  __int64 v56; // [rsp+128h] [rbp-198h]
  __int64 v57; // [rsp+130h] [rbp-190h]
  __int64 v58; // [rsp+138h] [rbp-188h]
  __int64 v59; // [rsp+140h] [rbp-180h]
  __int64 v60; // [rsp+148h] [rbp-178h]
  __int64 v61; // [rsp+150h] [rbp-170h]
  volatile signed __int32 *v62; // [rsp+158h] [rbp-168h] BYREF
  int v63; // [rsp+160h] [rbp-160h]
  __int64 v64[2]; // [rsp+168h] [rbp-158h] BYREF
  _QWORD v65[2]; // [rsp+178h] [rbp-148h] BYREF
  _QWORD v66[27]; // [rsp+188h] [rbp-138h] BYREF
  __int64 v67; // [rsp+260h] [rbp-60h]
  __int16 v68; // [rsp+268h] [rbp-58h]
  __int64 v69; // [rsp+270h] [rbp-50h]
  __int64 v70; // [rsp+278h] [rbp-48h]
  __int64 v71; // [rsp+280h] [rbp-40h]
  __int64 v72; // [rsp+288h] [rbp-38h]

  memset(v52, 0, 0x78u);
  result = sub_CCBB10((__m128i *)a3, v52);
  *(_QWORD *)(a3 + 4) = 0x300000000LL;
  if ( a1 > 0 )
  {
    v47 = a2;
    v46 = (__int64)&a2[(unsigned int)(a1 - 1) + 1];
    while ( 1 )
    {
      v5 = (const char *)*v47;
      if ( !memcmp(*v47, "-arch=compute_", 0xEu) )
      {
        v49 = v51;
        v6 = strlen(v5 + 14);
        sub_309D7C0((__int64 *)&v49, (_BYTE *)v5 + 14, (__int64)&v5[v6 + 14]);
        sub_222DF20((__int64)v66);
        v67 = 0;
        v68 = 0;
        v66[0] = off_4A06798;
        v69 = 0;
        v70 = 0;
        v71 = 0;
        v53 = (__int64 (__fastcall **)())qword_4A07108;
        v72 = 0;
        *(__int64 (__fastcall ***)())((char *)&v53 + qword_4A07108[-3]) = (__int64 (__fastcall **)())&unk_4A07130;
        v54 = 0;
        sub_222DD70((__int64)&v53 + (_QWORD)*(v53 - 3), 0);
        v56 = 0;
        v57 = 0;
        v58 = 0;
        v53 = off_4A07178;
        v66[0] = off_4A071A0;
        v55 = off_4A07480;
        v59 = 0;
        v60 = 0;
        v61 = 0;
        sub_220A990(&v62);
        v64[0] = (__int64)v65;
        v55 = off_4A07080;
        v63 = 0;
        sub_309D7C0(v64, v49, (__int64)v50 + (_QWORD)v49);
        v63 = 8;
        sub_223FD50((__int64)&v55, v64[0], 0, 0);
        sub_222DD70((__int64)v66, (__int64)&v55);
        sub_222E4D0((__int64 *)&v53, &v48, v7, v8);
        v9 = (_QWORD *)v64[0];
        v53 = off_4A07178;
        *(_DWORD *)a3 = 10 * v48;
        v66[0] = off_4A071A0;
        v55 = off_4A07080;
        if ( v9 == v65 )
          goto LABEL_13;
        goto LABEL_12;
      }
      if ( !memcmp(*v47, "-opt=", 5u) )
        break;
      if ( !memcmp(*v47, "-ftz=", 5u) )
      {
        v49 = v51;
        v16 = (__int64 (__fastcall **)())strlen(v5 + 5);
        v53 = v16;
        v17 = (size_t)v16;
        if ( (unsigned __int64)v16 > 0xF )
        {
          v49 = (_BYTE *)sub_22409D0((__int64)&v49, (unsigned __int64 *)&v53, 0);
          v36 = v49;
          v51[0] = v53;
        }
        else
        {
          if ( v16 == (__int64 (__fastcall **)())1 )
          {
            LOBYTE(v51[0]) = v5[5];
            v18 = v51;
LABEL_33:
            v50 = v16;
            *((_BYTE *)v16 + (_QWORD)v18) = 0;
            sub_222DF20((__int64)v66);
            v67 = 0;
            v66[0] = off_4A06798;
            v68 = 0;
            v69 = 0;
            v70 = 0;
            v53 = (__int64 (__fastcall **)())qword_4A07108;
            v71 = 0;
            v72 = 0;
            *(__int64 (__fastcall ***)())((char *)&v53 + qword_4A07108[-3]) = (__int64 (__fastcall **)())&unk_4A07130;
            v54 = 0;
            sub_222DD70((__int64)&v53 + (_QWORD)*(v53 - 3), 0);
            v56 = 0;
            v57 = 0;
            v58 = 0;
            v53 = off_4A07178;
            v66[0] = off_4A071A0;
            v55 = off_4A07480;
            v59 = 0;
            v60 = 0;
            v61 = 0;
            sub_220A990(&v62);
            v63 = 0;
            v55 = off_4A07080;
            v64[0] = (__int64)v65;
            sub_309D7C0(v64, v49, (__int64)v50 + (_QWORD)v49);
            v63 = 8;
            sub_223FD50((__int64)&v55, v64[0], 0, 0);
            sub_222DD70((__int64)v66, (__int64)&v55);
            sub_222E4D0((__int64 *)&v53, &v48, v19, v20);
            v21 = a3;
            v22 = 32 * (v48 & 1);
            v23 = *(_BYTE *)(a3 + 200) & 0xDF;
            goto LABEL_34;
          }
          if ( !v16 )
          {
            v18 = v51;
            goto LABEL_33;
          }
          v36 = v51;
        }
        memcpy(v36, v5 + 5, v17);
        v16 = v53;
        v18 = v49;
        goto LABEL_33;
      }
      if ( !memcmp(*v47, "-fma=", 5u) )
      {
        v49 = v51;
        v24 = (__int64 (__fastcall **)())strlen(v5 + 5);
        v53 = v24;
        v25 = (size_t)v24;
        if ( (unsigned __int64)v24 > 0xF )
        {
          v49 = (_BYTE *)sub_22409D0((__int64)&v49, (unsigned __int64 *)&v53, 0);
          v42 = v49;
          v51[0] = v53;
        }
        else
        {
          if ( v24 == (__int64 (__fastcall **)())1 )
          {
            LOBYTE(v51[0]) = v5[5];
            v26 = v51;
LABEL_40:
            v50 = v24;
            *((_BYTE *)v24 + (_QWORD)v26) = 0;
            sub_222DF20((__int64)v66);
            v67 = 0;
            v68 = 0;
            v66[0] = off_4A06798;
            v69 = 0;
            v70 = 0;
            v71 = 0;
            v53 = (__int64 (__fastcall **)())qword_4A07108;
            v72 = 0;
            *(__int64 (__fastcall ***)())((char *)&v53 + qword_4A07108[-3]) = (__int64 (__fastcall **)())&unk_4A07130;
            v54 = 0;
            sub_222DD70((__int64)&v53 + (_QWORD)*(v53 - 3), 0);
            v56 = 0;
            v57 = 0;
            v58 = 0;
            v53 = off_4A07178;
            v66[0] = off_4A071A0;
            v55 = off_4A07480;
            v59 = 0;
            v60 = 0;
            v61 = 0;
            sub_220A990(&v62);
            v63 = 0;
            v55 = off_4A07080;
            v64[0] = (__int64)v65;
            sub_309D7C0(v64, v49, (__int64)v50 + (_QWORD)v49);
            v63 = 8;
            sub_223FD50((__int64)&v55, v64[0], 0, 0);
            sub_222DD70((__int64)v66, (__int64)&v55);
            sub_222E4D0((__int64 *)&v53, &v48, v27, v28);
            v29 = *(_BYTE *)(a3 + 200);
            v53 = off_4A07178;
            v66[0] = off_4A071A0;
            *(_BYTE *)(a3 + 200) = ((_BYTE)v48 << 7) | v29 & 0x7F;
            goto LABEL_35;
          }
          if ( !v24 )
          {
            v26 = v51;
            goto LABEL_40;
          }
          v42 = v51;
        }
        memcpy(v42, v5 + 5, v25);
        v24 = v53;
        v26 = v49;
        goto LABEL_40;
      }
      if ( !memcmp(*v47, "-prec-div=", 0xAu) )
      {
        v49 = v51;
        v31 = (__int64 (__fastcall **)())strlen(v5 + 10);
        v53 = v31;
        v32 = (size_t)v31;
        if ( (unsigned __int64)v31 > 0xF )
        {
          v49 = (_BYTE *)sub_22409D0((__int64)&v49, (unsigned __int64 *)&v53, 0);
          v43 = v49;
          v51[0] = v53;
        }
        else
        {
          if ( v31 == (__int64 (__fastcall **)())1 )
          {
            LOBYTE(v51[0]) = v5[10];
            v33 = v51;
LABEL_48:
            v50 = v31;
            *((_BYTE *)v31 + (_QWORD)v33) = 0;
            sub_222DF20((__int64)v66);
            v68 = 0;
            v66[0] = off_4A06798;
            v67 = 0;
            v69 = 0;
            v70 = 0;
            v53 = (__int64 (__fastcall **)())qword_4A07108;
            v71 = 0;
            v72 = 0;
            *(__int64 (__fastcall ***)())((char *)&v53 + qword_4A07108[-3]) = (__int64 (__fastcall **)())&unk_4A07130;
            v54 = 0;
            sub_222DD70((__int64)&v53 + (_QWORD)*(v53 - 3), 0);
            v56 = 0;
            v53 = off_4A07178;
            v66[0] = off_4A071A0;
            v55 = off_4A07480;
            v57 = 0;
            v58 = 0;
            v59 = 0;
            v60 = 0;
            v61 = 0;
            sub_220A990(&v62);
            v64[0] = (__int64)v65;
            v55 = off_4A07080;
            v63 = 0;
            sub_309D7C0(v64, v49, (__int64)v50 + (_QWORD)v49);
            v63 = 8;
            sub_223FD50((__int64)&v55, v64[0], 0, 0);
            sub_222DD70((__int64)v66, (__int64)&v55);
            sub_222E4D0((__int64 *)&v53, &v48, v34, v35);
            v55 = off_4A07080;
            v15 = (_QWORD *)v64[0];
            v53 = off_4A07178;
            *(_DWORD *)(a3 + 204) = (v48 == 0) + 1;
            v66[0] = off_4A071A0;
            if ( v15 == v65 )
              goto LABEL_13;
LABEL_26:
            j_j___libc_free_0((unsigned __int64)v15);
            goto LABEL_13;
          }
          if ( !v31 )
          {
            v33 = v51;
            goto LABEL_48;
          }
          v43 = v51;
        }
        memcpy(v43, v5 + 10, v32);
        v31 = v53;
        v33 = v49;
        goto LABEL_48;
      }
      if ( !memcmp(*v47, "-prec-sqrt=", 0xBu) )
      {
        v49 = v51;
        v37 = (__int64 (__fastcall **)())strlen(v5 + 11);
        v53 = v37;
        v38 = (size_t)v37;
        if ( (unsigned __int64)v37 > 0xF )
        {
          v49 = (_BYTE *)sub_22409D0((__int64)&v49, (unsigned __int64 *)&v53, 0);
          v44 = v49;
          v51[0] = v53;
        }
        else
        {
          if ( v37 == (__int64 (__fastcall **)())1 )
          {
            LOBYTE(v51[0]) = v5[11];
            v39 = v51;
            goto LABEL_63;
          }
          if ( !v37 )
          {
            v39 = v51;
            goto LABEL_63;
          }
          v44 = v51;
        }
        memcpy(v44, v5 + 11, v38);
        v37 = v53;
        v39 = v49;
LABEL_63:
        v50 = v37;
        *((_BYTE *)v37 + (_QWORD)v39) = 0;
        sub_222DF20((__int64)v66);
        v67 = 0;
        v69 = 0;
        v66[0] = off_4A06798;
        v68 = 0;
        v70 = 0;
        v71 = 0;
        v72 = 0;
        v53 = (__int64 (__fastcall **)())qword_4A07108;
        *(__int64 (__fastcall ***)())((char *)&v53 + qword_4A07108[-3]) = (__int64 (__fastcall **)())&unk_4A07130;
        v54 = 0;
        sub_222DD70((__int64)&v53 + (_QWORD)*(v53 - 3), 0);
        v56 = 0;
        v57 = 0;
        v58 = 0;
        v53 = off_4A07178;
        v66[0] = off_4A071A0;
        v55 = off_4A07480;
        v59 = 0;
        v60 = 0;
        v61 = 0;
        sub_220A990(&v62);
        v63 = 0;
        v55 = off_4A07080;
        v64[0] = (__int64)v65;
        sub_309D7C0(v64, v49, (__int64)v50 + (_QWORD)v49);
        v63 = 8;
        sub_223FD50((__int64)&v55, v64[0], 0, 0);
        sub_222DD70((__int64)v66, (__int64)&v55);
        sub_222E4D0((__int64 *)&v53, &v48, v40, v41);
        v21 = a3;
        v23 = *(_BYTE *)(a3 + 200) & 0xBF;
        v22 = (v48 == 0) << 6;
LABEL_34:
        *(_BYTE *)(v21 + 200) = v22 | v23;
        v53 = off_4A07178;
        v66[0] = off_4A071A0;
LABEL_35:
        v9 = (_QWORD *)v64[0];
        v55 = off_4A07080;
        if ( (_QWORD *)v64[0] == v65 )
        {
LABEL_13:
          v55 = off_4A07480;
          sub_2209150(&v62);
          v53 = (__int64 (__fastcall **)())qword_4A07108;
          *(__int64 (__fastcall ***)())((char *)&v53 + qword_4A07108[-3]) = (__int64 (__fastcall **)())&unk_4A07130;
          v54 = 0;
          v66[0] = off_4A06798;
          sub_222E050((__int64)v66);
          if ( v49 != (_BYTE *)v51 )
            j_j___libc_free_0((unsigned __int64)v49);
          goto LABEL_15;
        }
LABEL_12:
        j_j___libc_free_0((unsigned __int64)v9);
        goto LABEL_13;
      }
      if ( !strcmp((const char *)*v47, "--device-c") )
      {
        *(_DWORD *)(a3 + 4) = 2;
      }
      else if ( *v5 == 45 && v5[1] == 103 && !v5[2] )
      {
        *(_DWORD *)(a3 + 12) = 2;
      }
      else if ( !strcmp((const char *)*v47, "-generate-line-info") )
      {
        *(_DWORD *)(a3 + 12) = 1;
      }
LABEL_15:
      result = (__int64)++v47;
      if ( v47 == (const void **)v46 )
        return result;
    }
    v49 = v51;
    v10 = (__int64 (__fastcall **)())strlen(v5 + 5);
    v53 = v10;
    v11 = (size_t)v10;
    if ( (unsigned __int64)v10 > 0xF )
    {
      v49 = (_BYTE *)sub_22409D0((__int64)&v49, (unsigned __int64 *)&v53, 0);
      v30 = v49;
      v51[0] = v53;
    }
    else
    {
      if ( v10 == (__int64 (__fastcall **)())1 )
      {
        LOBYTE(v51[0]) = v5[5];
        v12 = v51;
        goto LABEL_20;
      }
      if ( !v10 )
      {
        v12 = v51;
        goto LABEL_20;
      }
      v30 = v51;
    }
    memcpy(v30, v5 + 5, v11);
    v10 = v53;
    v12 = v49;
LABEL_20:
    v50 = v10;
    *((_BYTE *)v10 + (_QWORD)v12) = 0;
    sub_222DF20((__int64)v66);
    v67 = 0;
    v68 = 0;
    v66[0] = off_4A06798;
    v69 = 0;
    v70 = 0;
    v71 = 0;
    v53 = (__int64 (__fastcall **)())qword_4A07108;
    v72 = 0;
    *(__int64 (__fastcall ***)())((char *)&v53 + qword_4A07108[-3]) = (__int64 (__fastcall **)())&unk_4A07130;
    v54 = 0;
    sub_222DD70((__int64)&v53 + (_QWORD)*(v53 - 3), 0);
    v56 = 0;
    v57 = 0;
    v58 = 0;
    v53 = off_4A07178;
    v66[0] = off_4A071A0;
    v59 = 0;
    v55 = off_4A07480;
    v60 = 0;
    v61 = 0;
    sub_220A990(&v62);
    v64[0] = (__int64)v65;
    v55 = off_4A07080;
    v63 = 0;
    sub_309D7C0(v64, v49, (__int64)v50 + (_QWORD)v49);
    v63 = 8;
    sub_223FD50((__int64)&v55, v64[0], 0, 0);
    sub_222DD70((__int64)v66, (__int64)&v55);
    sub_222E4D0((__int64 *)&v53, &v48, v13, v14);
    if ( v48 == 2 )
    {
      *(_DWORD *)(a3 + 8) = 2;
    }
    else if ( v48 > 2 )
    {
      if ( v48 == 3 )
        *(_DWORD *)(a3 + 8) = 3;
    }
    else if ( v48 )
    {
      if ( v48 == 1 )
        *(_DWORD *)(a3 + 8) = 1;
    }
    else
    {
      *(_DWORD *)(a3 + 8) = 0;
    }
    v15 = (_QWORD *)v64[0];
    v53 = off_4A07178;
    v66[0] = off_4A071A0;
    v55 = off_4A07080;
    if ( (_QWORD *)v64[0] == v65 )
      goto LABEL_13;
    goto LABEL_26;
  }
  return result;
}
