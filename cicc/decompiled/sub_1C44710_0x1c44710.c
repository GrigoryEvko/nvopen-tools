// Function: sub_1C44710
// Address: 0x1c44710
//
_BOOL8 __fastcall sub_1C44710(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v4; // r12d
  char v5; // r13
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  int v8; // eax
  unsigned __int64 v9; // r13
  __int64 v10; // rdx
  unsigned __int64 v11; // r14
  unsigned __int64 v12; // r12
  __int64 v13; // rbx
  unsigned __int64 v14; // r15
  _QWORD *v15; // rbx
  unsigned __int64 v16; // r12
  _QWORD *v17; // rbx
  _BYTE *v19; // r13
  size_t v20; // r12
  _QWORD *v21; // rax
  _BYTE *v22; // rsi
  _BYTE *v23; // rsi
  __int64 v24; // rax
  bool v25; // bl
  __int64 v26; // r13
  int v27; // edx
  __int64 v28; // rax
  char v29; // r12
  __int64 v30; // rcx
  unsigned __int64 v31; // rbx
  unsigned __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdx
  unsigned __int64 v35; // rsi
  unsigned __int64 v36; // rsi
  __int64 v37; // rdx
  __int64 *v38; // rax
  __int64 v39; // rsi
  unsigned __int64 v40; // rdi
  __int64 v41; // rsi
  _QWORD *v42; // rdi
  _BYTE *v43; // r13
  size_t v44; // r12
  size_t v45; // r15
  __int64 **v46; // rax
  _QWORD *v47; // rax
  bool v48; // zf
  __int64 v49; // rcx
  unsigned __int64 v50; // rsi
  __int64 v51; // rcx
  unsigned int v52; // [rsp+48h] [rbp-5C8h]
  __int64 v53; // [rsp+58h] [rbp-5B8h]
  __int64 v54; // [rsp+68h] [rbp-5A8h]
  char v56; // [rsp+78h] [rbp-598h]
  bool v57; // [rsp+84h] [rbp-58Ch]
  char v58; // [rsp+84h] [rbp-58Ch]
  unsigned __int64 v59; // [rsp+88h] [rbp-588h]
  _BYTE *v60; // [rsp+88h] [rbp-588h]
  int v61; // [rsp+9Ch] [rbp-574h] BYREF
  __int64 v62; // [rsp+A0h] [rbp-570h] BYREF
  _BYTE *v63; // [rsp+A8h] [rbp-568h]
  _BYTE *v64; // [rsp+B0h] [rbp-560h]
  __int64 v65; // [rsp+C0h] [rbp-550h] BYREF
  _BYTE *v66; // [rsp+C8h] [rbp-548h]
  _BYTE *v67; // [rsp+D0h] [rbp-540h]
  _QWORD *v68; // [rsp+E0h] [rbp-530h] BYREF
  unsigned __int64 v69; // [rsp+E8h] [rbp-528h]
  _QWORD v70[2]; // [rsp+F0h] [rbp-520h] BYREF
  _QWORD *v71; // [rsp+100h] [rbp-510h] BYREF
  __int64 v72; // [rsp+108h] [rbp-508h]
  _QWORD v73[2]; // [rsp+110h] [rbp-500h] BYREF
  _QWORD *v74; // [rsp+120h] [rbp-4F0h] BYREF
  __int64 v75; // [rsp+128h] [rbp-4E8h]
  _QWORD v76[2]; // [rsp+130h] [rbp-4E0h] BYREF
  __int64 (__fastcall **v77)(); // [rsp+140h] [rbp-4D0h] BYREF
  __int64 v78; // [rsp+148h] [rbp-4C8h]
  __int64 (__fastcall **v79)(); // [rsp+150h] [rbp-4C0h] BYREF
  _QWORD v80[3]; // [rsp+158h] [rbp-4B8h] BYREF
  unsigned __int64 v81; // [rsp+170h] [rbp-4A0h]
  __int64 v82; // [rsp+178h] [rbp-498h]
  unsigned __int64 v83; // [rsp+180h] [rbp-490h]
  __int64 v84; // [rsp+188h] [rbp-488h]
  char v85[8]; // [rsp+190h] [rbp-480h] BYREF
  int v86; // [rsp+198h] [rbp-478h]
  _QWORD v87[2]; // [rsp+1A0h] [rbp-470h] BYREF
  _QWORD v88[2]; // [rsp+1B0h] [rbp-460h] BYREF
  _QWORD v89[28]; // [rsp+1C0h] [rbp-450h] BYREF
  __int16 v90; // [rsp+2A0h] [rbp-370h]
  __int64 v91; // [rsp+2A8h] [rbp-368h]
  __int64 v92; // [rsp+2B0h] [rbp-360h]
  __int64 v93; // [rsp+2B8h] [rbp-358h]
  __int64 v94; // [rsp+2C0h] [rbp-350h]
  unsigned __int64 v95; // [rsp+2D0h] [rbp-340h] BYREF
  unsigned int v96; // [rsp+2D8h] [rbp-338h]
  char v97; // [rsp+2E0h] [rbp-330h] BYREF

  v2 = *(_QWORD *)(a2 - 24);
  v53 = v2;
  if ( *(_BYTE *)(v2 + 16) != 20 )
    BUG();
  sub_15F1410(&v95, *(_BYTE **)(v2 + 56), *(_QWORD *)(v2 + 64));
  v4 = v96;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v61 = 0;
  LODWORD(v77) = 0;
  if ( !v96 )
  {
    v57 = 0;
    goto LABEL_15;
  }
  v5 = 0;
  v6 = 0;
  do
  {
    v7 = v95 + 192 * v6;
    if ( *(_DWORD *)(v7 + 24) == 1 && !*(_DWORD *)v7 )
    {
      if ( (unsigned int)sub_2241AC0(*(_QWORD *)(v7 + 16), "N") )
      {
        v8 = v61;
      }
      else
      {
        v22 = v63;
        if ( v63 == v64 )
        {
          sub_B8BBF0((__int64)&v62, v63, &v77);
        }
        else
        {
          if ( v63 )
          {
            *(_DWORD *)v63 = (_DWORD)v77;
            v22 = v63;
          }
          v63 = v22 + 4;
        }
        v23 = v66;
        if ( v66 == v67 )
        {
          v5 = 1;
          sub_B8BBF0((__int64)&v65, v66, &v61);
          v8 = v61;
        }
        else
        {
          v8 = v61;
          if ( v66 )
          {
            *(_DWORD *)v66 = v61;
            v23 = v66;
          }
          v5 = 1;
          v66 = v23 + 4;
        }
      }
      v61 = v8 + 1;
    }
    v6 = (unsigned int)((_DWORD)v77 + 1);
    LODWORD(v77) = v6;
  }
  while ( (_DWORD)v6 != v4 );
  v57 = v5;
  if ( v5 )
  {
    v68 = v70;
    v19 = *(_BYTE **)(v53 + 24);
    v20 = *(_QWORD *)(v53 + 32);
    v57 = v19 == 0 && &v19[v20] != 0;
    if ( v57 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v77 = *(__int64 (__fastcall ***)())(v53 + 32);
    if ( v20 > 0xF )
    {
      v68 = (_QWORD *)sub_22409D0(&v68, &v77, 0);
      v42 = v68;
      v70[0] = v77;
    }
    else
    {
      if ( v20 == 1 )
      {
        LOBYTE(v70[0]) = *v19;
        v21 = v70;
LABEL_54:
        v69 = v20;
        *((_BYTE *)v21 + v20) = 0;
        v24 = (__int64)&v63[-v62] >> 2;
        if ( !(_DWORD)v24 )
          goto LABEL_85;
        v25 = 0;
        v26 = 0;
        v54 = 4LL * (unsigned int)v24;
        do
        {
          v27 = *(_DWORD *)(a2 + 20);
          v72 = 0;
          LOBYTE(v73[0]) = 0;
          v71 = v73;
          v28 = sub_1649C60(*(_QWORD *)(a2 + 24 * (*(unsigned int *)(v65 + v26) - (unsigned __int64)(v27 & 0xFFFFFFF))));
          v29 = sub_1C43660((__int64)a1, v28, (__int64)&v71);
          if ( v29 )
          {
            v52 = *(_DWORD *)(v62 + v26);
            sub_222DF20(v89);
            v89[27] = 0;
            v90 = 0;
            v91 = 0;
            v89[0] = off_4A06798;
            v92 = 0;
            v93 = 0;
            v94 = 0;
            v77 = (__int64 (__fastcall **)())qword_4A072D8;
            *(__int64 (__fastcall ***)())((char *)&v77 + qword_4A072D8[-3]) = (__int64 (__fastcall **)())&unk_4A07300;
            v78 = 0;
            sub_222DD70((char *)&v77 + (_QWORD)*(v77 - 3), 0);
            v79 = (__int64 (__fastcall **)())qword_4A07288;
            *(_QWORD *)((char *)&v80[-1] + qword_4A07288[-3]) = &unk_4A072B0;
            sub_222DD70((char *)&v80[-1] + (_QWORD)*(v79 - 3), 0);
            v77 = (__int64 (__fastcall **)())qword_4A07328;
            *(__int64 (__fastcall ***)())((char *)&v77 + qword_4A07328[-3]) = (__int64 (__fastcall **)())&unk_4A07378;
            v77 = off_4A073F0;
            v89[0] = off_4A07440;
            v79 = off_4A07418;
            v80[0] = off_4A07480;
            v80[1] = 0;
            v80[2] = 0;
            v81 = 0;
            v82 = 0;
            v83 = 0;
            v84 = 0;
            sub_220A990(v85);
            v86 = 24;
            v87[1] = 0;
            v87[0] = v88;
            v80[0] = off_4A07080;
            LOBYTE(v88[0]) = 0;
            sub_222DD70(v89, v80);
            sub_223E0D0(&v79, "$", 1);
            sub_223E760(&v79, v52);
            v75 = 0;
            v74 = v76;
            LOBYTE(v76[0]) = 0;
            if ( v83 )
            {
              if ( v83 <= v81 )
                sub_2241130(&v74, 0, 0, v82, v81 - v82);
              else
                sub_2241130(&v74, 0, 0, v82, v83 - v82);
            }
            else
            {
              sub_2240AE0(&v74, v87);
            }
            v30 = v75;
            v31 = (unsigned int)v75;
            while ( 1 )
            {
              v33 = sub_22416F0(&v68, v74, 0, v30);
              v35 = v33;
              if ( v33 == -1 )
                break;
              v32 = v69 - v33;
              if ( v31 <= v69 - v35 )
                v32 = v31;
              if ( v35 > v69 )
                sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::replace");
              sub_2241130(&v68, v35, v32, v71, v72);
              v30 = v75;
            }
            if ( v74 != v76 )
            {
              v35 = v76[0] + 1LL;
              j_j___libc_free_0(v74, v76[0] + 1LL);
            }
            v77 = off_4A073F0;
            v89[0] = off_4A07440;
            v79 = off_4A07418;
            v80[0] = off_4A07080;
            if ( (_QWORD *)v87[0] != v88 )
            {
              v35 = v88[0] + 1LL;
              j_j___libc_free_0(v87[0], v88[0] + 1LL);
            }
            v80[0] = off_4A07480;
            sub_2209150(v85, v35, v34);
            v77 = (__int64 (__fastcall **)())qword_4A07328;
            *(__int64 (__fastcall ***)())((char *)&v77 + qword_4A07328[-3]) = (__int64 (__fastcall **)())&unk_4A07378;
            v79 = (__int64 (__fastcall **)())qword_4A07288;
            *(_QWORD *)((char *)&v80[-1] + qword_4A07288[-3]) = &unk_4A072B0;
            v77 = (__int64 (__fastcall **)())qword_4A072D8;
            *(__int64 (__fastcall ***)())((char *)&v77 + qword_4A072D8[-3]) = (__int64 (__fastcall **)())&unk_4A07300;
            v78 = 0;
            v89[0] = off_4A06798;
            sub_222E050(v89);
            v36 = *(_QWORD *)(a2
                            + 24 * (*(unsigned int *)(v65 + v26) - (unsigned __int64)(*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
            if ( *(_BYTE *)(v36 + 16) > 0x17u )
            {
              sub_1C44570(a1, v36);
              v36 = *(_QWORD *)(a2
                              + 24
                              * (*(unsigned int *)(v65 + v26) - (unsigned __int64)(*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
            }
            v37 = sub_1599EF0(*(__int64 ***)v36);
            v38 = (__int64 *)(a2
                            + 24 * (*(unsigned int *)(v65 + v26) - (unsigned __int64)(*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
            if ( *v38 )
            {
              v39 = v38[1];
              v40 = v38[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v40 = v39;
              if ( v39 )
                *(_QWORD *)(v39 + 16) = v40 | *(_QWORD *)(v39 + 16) & 3LL;
            }
            *v38 = v37;
            v25 = v29;
            if ( v37 )
            {
              v41 = *(_QWORD *)(v37 + 8);
              v38[1] = v41;
              if ( v41 )
                *(_QWORD *)(v41 + 16) = (unsigned __int64)(v38 + 1) | *(_QWORD *)(v41 + 16) & 3LL;
              v25 = v29;
              v38[2] = (v37 + 8) | v38[2] & 3;
              *(_QWORD *)(v37 + 8) = v38;
            }
          }
          if ( v71 != v73 )
            j_j___libc_free_0(v71, v73[0] + 1LL);
          v26 += 4;
        }
        while ( v54 != v26 );
        if ( v25 )
        {
          v43 = v68;
          v44 = v69;
          v45 = *(_QWORD *)(v53 + 64);
          v56 = *(_BYTE *)(v53 + 97);
          v58 = *(_BYTE *)(v53 + 96);
          v60 = *(_BYTE **)(v53 + 56);
          v46 = (__int64 **)sub_15EAB70(v53);
          v47 = (_QWORD *)sub_15EE570(v46, v43, v44, v60, v45, v58, v56, 0);
          v48 = *(_QWORD *)(a2 - 24) == 0;
          *(_QWORD *)(a2 + 64) = *(_QWORD *)(*v47 + 24LL);
          if ( !v48 )
          {
            v49 = *(_QWORD *)(a2 - 16);
            v50 = *(_QWORD *)(a2 - 8) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v50 = v49;
            if ( v49 )
              *(_QWORD *)(v49 + 16) = v50 | *(_QWORD *)(v49 + 16) & 3LL;
          }
          *(_QWORD *)(a2 - 24) = v47;
          v51 = v47[1];
          *(_QWORD *)(a2 - 16) = v51;
          if ( v51 )
            *(_QWORD *)(v51 + 16) = (a2 - 16) | *(_QWORD *)(v51 + 16) & 3LL;
          *(_QWORD *)(a2 - 8) = *(_QWORD *)(a2 - 8) & 3LL | (unsigned __int64)(v47 + 1);
          v47[1] = a2 - 24;
          if ( v68 != v70 )
            j_j___libc_free_0(v68, v70[0] + 1LL);
          v57 = v25;
        }
        else
        {
LABEL_85:
          if ( v68 != v70 )
            j_j___libc_free_0(v68, v70[0] + 1LL);
        }
        goto LABEL_11;
      }
      if ( !v20 )
      {
        v21 = v70;
        goto LABEL_54;
      }
      v42 = v70;
    }
    memcpy(v42, v19, v20);
    v20 = (size_t)v77;
    v21 = v68;
    goto LABEL_54;
  }
LABEL_11:
  if ( v65 )
    j_j___libc_free_0(v65, &v67[-v65]);
  if ( v62 )
    j_j___libc_free_0(v62, &v64[-v62]);
LABEL_15:
  v59 = v95;
  v9 = v95 + 192LL * v96;
  if ( v95 != v9 )
  {
    do
    {
      v10 = *(unsigned int *)(v9 - 120);
      v11 = *(_QWORD *)(v9 - 128);
      v9 -= 192LL;
      v12 = v11 + 56 * v10;
      if ( v11 != v12 )
      {
        do
        {
          v13 = *(unsigned int *)(v12 - 40);
          v14 = *(_QWORD *)(v12 - 48);
          v12 -= 56LL;
          v15 = (_QWORD *)(v14 + 32 * v13);
          if ( (_QWORD *)v14 != v15 )
          {
            do
            {
              v15 -= 4;
              if ( (_QWORD *)*v15 != v15 + 2 )
                j_j___libc_free_0(*v15, v15[2] + 1LL);
            }
            while ( (_QWORD *)v14 != v15 );
            v14 = *(_QWORD *)(v12 + 8);
          }
          if ( v14 != v12 + 24 )
            _libc_free(v14);
        }
        while ( v11 != v12 );
        v11 = *(_QWORD *)(v9 + 64);
      }
      if ( v11 != v9 + 80 )
        _libc_free(v11);
      v16 = *(_QWORD *)(v9 + 16);
      v17 = (_QWORD *)(v16 + 32LL * *(unsigned int *)(v9 + 24));
      if ( (_QWORD *)v16 != v17 )
      {
        do
        {
          v17 -= 4;
          if ( (_QWORD *)*v17 != v17 + 2 )
            j_j___libc_free_0(*v17, v17[2] + 1LL);
        }
        while ( (_QWORD *)v16 != v17 );
        v16 = *(_QWORD *)(v9 + 16);
      }
      if ( v16 != v9 + 32 )
        _libc_free(v16);
    }
    while ( v59 != v9 );
    v9 = v95;
  }
  if ( (char *)v9 != &v97 )
    _libc_free(v9);
  return v57;
}
