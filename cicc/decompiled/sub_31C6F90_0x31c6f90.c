// Function: sub_31C6F90
// Address: 0x31c6f90
//
_BOOL8 __fastcall sub_31C6F90(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int8 *v4; // rsi
  unsigned int v5; // r12d
  char v6; // r13
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  int v9; // eax
  unsigned __int64 v10; // r13
  __int64 v11; // rdx
  unsigned __int64 v12; // r14
  unsigned __int64 v13; // r12
  __int64 v14; // rbx
  unsigned __int64 v15; // r15
  unsigned __int64 *v16; // rbx
  unsigned __int64 v17; // r12
  unsigned __int64 *v18; // rbx
  unsigned __int8 *v20; // r13
  size_t v21; // r12
  _QWORD *v22; // rax
  _BYTE *v23; // rsi
  __int64 v24; // rax
  bool v25; // bl
  __int64 v26; // r13
  int v27; // edx
  char v28; // r12
  size_t v29; // rcx
  unsigned __int64 v30; // rbx
  unsigned __int64 v31; // rax
  __int64 v32; // rax
  size_t v33; // rsi
  __int64 v34; // rdx
  __int64 *v35; // rax
  __int64 v36; // rcx
  __int64 v37; // rcx
  _QWORD *v38; // rdi
  __int64 v39; // r12
  __int64 v40; // r13
  __int64 v41; // r15
  _QWORD **v42; // rax
  __int64 v43; // rax
  __int64 v44; // r12
  __int64 v45; // rax
  bool v46; // zf
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  unsigned int v50; // [rsp+48h] [rbp-5C8h]
  __int64 v51; // [rsp+58h] [rbp-5B8h]
  __int64 v52; // [rsp+68h] [rbp-5A8h]
  char v54; // [rsp+78h] [rbp-598h]
  bool v55; // [rsp+84h] [rbp-58Ch]
  char v56; // [rsp+84h] [rbp-58Ch]
  __int64 v57; // [rsp+88h] [rbp-588h]
  __int64 v58; // [rsp+88h] [rbp-588h]
  int v59; // [rsp+9Ch] [rbp-574h] BYREF
  unsigned __int64 v60; // [rsp+A0h] [rbp-570h] BYREF
  _BYTE *v61; // [rsp+A8h] [rbp-568h]
  _BYTE *v62; // [rsp+B0h] [rbp-560h]
  unsigned __int64 v63; // [rsp+C0h] [rbp-550h] BYREF
  unsigned __int8 *v64; // [rsp+C8h] [rbp-548h]
  unsigned __int8 *v65; // [rsp+D0h] [rbp-540h]
  _QWORD *v66; // [rsp+E0h] [rbp-530h] BYREF
  size_t v67; // [rsp+E8h] [rbp-528h]
  _QWORD v68[2]; // [rsp+F0h] [rbp-520h] BYREF
  _BYTE *v69; // [rsp+100h] [rbp-510h] BYREF
  size_t v70; // [rsp+108h] [rbp-508h]
  _QWORD v71[2]; // [rsp+110h] [rbp-500h] BYREF
  char *v72; // [rsp+120h] [rbp-4F0h] BYREF
  size_t v73; // [rsp+128h] [rbp-4E8h]
  _BYTE v74[16]; // [rsp+130h] [rbp-4E0h] BYREF
  __int64 (__fastcall **v75)(); // [rsp+140h] [rbp-4D0h] BYREF
  __int64 v76; // [rsp+148h] [rbp-4C8h]
  __int64 (__fastcall **v77)(); // [rsp+150h] [rbp-4C0h] BYREF
  _QWORD v78[3]; // [rsp+158h] [rbp-4B8h] BYREF
  unsigned __int64 v79; // [rsp+170h] [rbp-4A0h]
  _BYTE *v80; // [rsp+178h] [rbp-498h]
  unsigned __int64 v81; // [rsp+180h] [rbp-490h]
  __int64 v82; // [rsp+188h] [rbp-488h]
  volatile signed __int32 *v83; // [rsp+190h] [rbp-480h] BYREF
  int v84; // [rsp+198h] [rbp-478h]
  unsigned __int64 v85[2]; // [rsp+1A0h] [rbp-470h] BYREF
  _BYTE v86[16]; // [rsp+1B0h] [rbp-460h] BYREF
  _QWORD v87[28]; // [rsp+1C0h] [rbp-450h] BYREF
  __int16 v88; // [rsp+2A0h] [rbp-370h]
  __int64 v89; // [rsp+2A8h] [rbp-368h]
  __int64 v90; // [rsp+2B0h] [rbp-360h]
  __int64 v91; // [rsp+2B8h] [rbp-358h]
  __int64 v92; // [rsp+2C0h] [rbp-350h]
  unsigned __int64 v93; // [rsp+2D0h] [rbp-340h] BYREF
  unsigned int v94; // [rsp+2D8h] [rbp-338h]
  char v95; // [rsp+2E0h] [rbp-330h] BYREF

  v2 = *(_QWORD *)(a2 - 32);
  v51 = v2;
  if ( *(_BYTE *)v2 != 25 )
    BUG();
  v4 = *(unsigned __int8 **)(v2 + 56);
  sub_B428A0((__int64 *)&v93, v4, *(_QWORD *)(v2 + 64));
  v5 = v94;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v59 = 0;
  LODWORD(v75) = 0;
  if ( !v94 )
  {
    v55 = 0;
    goto LABEL_15;
  }
  v6 = 0;
  v7 = 0;
  do
  {
    v8 = v93 + 192 * v7;
    if ( *(_DWORD *)(v8 + 24) == 1 && !*(_DWORD *)v8 )
    {
      v4 = (unsigned __int8 *)"N";
      if ( sub_2241AC0(*(_QWORD *)(v8 + 16), "N") )
      {
        v9 = v59;
      }
      else
      {
        v23 = v61;
        if ( v61 == v62 )
        {
          sub_B8BBF0((__int64)&v60, v61, &v75);
        }
        else
        {
          if ( v61 )
          {
            *(_DWORD *)v61 = (_DWORD)v75;
            v23 = v61;
          }
          v61 = v23 + 4;
        }
        v4 = v64;
        if ( v64 == v65 )
        {
          v6 = 1;
          sub_B8BBF0((__int64)&v63, v64, &v59);
          v9 = v59;
        }
        else
        {
          v9 = v59;
          if ( v64 )
          {
            *(_DWORD *)v64 = v59;
            v4 = v64;
          }
          v4 += 4;
          v6 = 1;
          v64 = v4;
        }
      }
      v59 = v9 + 1;
    }
    v7 = (unsigned int)((_DWORD)v75 + 1);
    LODWORD(v75) = v7;
  }
  while ( (_DWORD)v7 != v5 );
  v55 = v6;
  if ( v6 )
  {
    v66 = v68;
    v20 = *(unsigned __int8 **)(v51 + 24);
    v21 = *(_QWORD *)(v51 + 32);
    v55 = v20 == 0 && &v20[v21] != 0;
    if ( v55 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v75 = *(__int64 (__fastcall ***)())(v51 + 32);
    if ( v21 > 0xF )
    {
      v66 = (_QWORD *)sub_22409D0((__int64)&v66, (unsigned __int64 *)&v75, 0);
      v38 = v66;
      v68[0] = v75;
    }
    else
    {
      if ( v21 == 1 )
      {
        LOBYTE(v68[0]) = *v20;
        v22 = v68;
LABEL_54:
        v67 = v21;
        *((_BYTE *)v22 + v21) = 0;
        v24 = (__int64)&v61[-v60] >> 2;
        if ( !(_DWORD)v24 )
          goto LABEL_85;
        v25 = 0;
        v26 = 0;
        v52 = 4LL * (unsigned int)v24;
        do
        {
          v27 = *(_DWORD *)(a2 + 4);
          v70 = 0;
          LOBYTE(v71[0]) = 0;
          v69 = v71;
          v4 = sub_BD3990(
                 *(unsigned __int8 **)(a2 + 32 * (*(unsigned int *)(v63 + v26) - (unsigned __int64)(v27 & 0x7FFFFFF))),
                 (__int64)v4);
          v28 = sub_31C63D0((__int64)a1, v4, (__int64)&v69);
          if ( v28 )
          {
            v50 = *(_DWORD *)(v60 + v26);
            sub_222DF20((__int64)v87);
            v87[27] = 0;
            v89 = 0;
            v90 = 0;
            v87[0] = off_4A06798;
            v88 = 0;
            v91 = 0;
            v92 = 0;
            v75 = (__int64 (__fastcall **)())qword_4A072D8;
            *(__int64 (__fastcall ***)())((char *)&v75 + qword_4A072D8[-3]) = (__int64 (__fastcall **)())&unk_4A07300;
            v76 = 0;
            sub_222DD70((__int64)&v75 + (_QWORD)*(v75 - 3), 0);
            v77 = (__int64 (__fastcall **)())qword_4A07288;
            *(_QWORD *)((char *)&v78[-1] + qword_4A07288[-3]) = &unk_4A072B0;
            sub_222DD70((__int64)&v78[-1] + (_QWORD)*(v77 - 3), 0);
            v75 = (__int64 (__fastcall **)())qword_4A07328;
            *(__int64 (__fastcall ***)())((char *)&v75 + qword_4A07328[-3]) = (__int64 (__fastcall **)())&unk_4A07378;
            v78[1] = 0;
            v75 = off_4A073F0;
            v87[0] = off_4A07440;
            v77 = off_4A07418;
            v78[0] = off_4A07480;
            v78[2] = 0;
            v79 = 0;
            v80 = 0;
            v81 = 0;
            v82 = 0;
            sub_220A990(&v83);
            v85[0] = (unsigned __int64)v86;
            v78[0] = off_4A07080;
            v84 = 24;
            v85[1] = 0;
            v86[0] = 0;
            sub_222DD70((__int64)v87, (__int64)v78);
            sub_223E0D0((__int64 *)&v77, "$", 1);
            sub_223E760((__int64 *)&v77, v50);
            v73 = 0;
            v72 = v74;
            v74[0] = 0;
            if ( v81 )
            {
              if ( v81 <= v79 )
                sub_2241130((unsigned __int64 *)&v72, 0, 0, v80, v79 - (_QWORD)v80);
              else
                sub_2241130((unsigned __int64 *)&v72, 0, 0, v80, v81 - (_QWORD)v80);
            }
            else
            {
              sub_2240AE0((unsigned __int64 *)&v72, v85);
            }
            v29 = v73;
            v30 = (unsigned int)v73;
            while ( 1 )
            {
              v32 = sub_22416F0((__int64 *)&v66, v72, 0, v29);
              v33 = v32;
              if ( v32 == -1 )
                break;
              v31 = v67 - v32;
              if ( v30 <= v67 - v33 )
                v31 = v30;
              if ( v33 > v67 )
                sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", "basic_string::replace", v33, v67);
              sub_2241130((unsigned __int64 *)&v66, v33, v31, v69, v70);
              v29 = v73;
            }
            if ( v72 != v74 )
              j_j___libc_free_0((unsigned __int64)v72);
            v75 = off_4A073F0;
            v87[0] = off_4A07440;
            v77 = off_4A07418;
            v78[0] = off_4A07080;
            if ( (_BYTE *)v85[0] != v86 )
              j_j___libc_free_0(v85[0]);
            v78[0] = off_4A07480;
            sub_2209150(&v83);
            v75 = (__int64 (__fastcall **)())qword_4A07328;
            *(__int64 (__fastcall ***)())((char *)&v75 + qword_4A07328[-3]) = (__int64 (__fastcall **)())&unk_4A07378;
            v77 = (__int64 (__fastcall **)())qword_4A07288;
            *(_QWORD *)((char *)&v78[-1] + qword_4A07288[-3]) = &unk_4A072B0;
            v75 = (__int64 (__fastcall **)())qword_4A072D8;
            *(__int64 (__fastcall ***)())((char *)&v75 + qword_4A072D8[-3]) = (__int64 (__fastcall **)())&unk_4A07300;
            v76 = 0;
            v87[0] = off_4A06798;
            sub_222E050((__int64)v87);
            v4 = *(unsigned __int8 **)(a2
                                     + 32
                                     * (*(unsigned int *)(v63 + v26)
                                      - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
            if ( *v4 > 0x1Cu )
            {
              sub_31C6E20(a1, (unsigned __int64)v4);
              v4 = *(unsigned __int8 **)(a2
                                       + 32
                                       * (*(unsigned int *)(v63 + v26)
                                        - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
            }
            v34 = sub_ACA8A0(*((__int64 ***)v4 + 1));
            v35 = (__int64 *)(a2
                            + 32 * (*(unsigned int *)(v63 + v26) - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
            if ( *v35 )
            {
              v4 = (unsigned __int8 *)v35[2];
              v36 = v35[1];
              *(_QWORD *)v4 = v36;
              if ( v36 )
              {
                v4 = (unsigned __int8 *)v35[2];
                *(_QWORD *)(v36 + 16) = v4;
              }
            }
            *v35 = v34;
            v25 = v28;
            if ( v34 )
            {
              v37 = *(_QWORD *)(v34 + 16);
              v4 = (unsigned __int8 *)(v34 + 16);
              v35[1] = v37;
              if ( v37 )
                *(_QWORD *)(v37 + 16) = v35 + 1;
              v35[2] = (__int64)v4;
              v25 = v28;
              *(_QWORD *)(v34 + 16) = v35;
            }
          }
          if ( v69 != (_BYTE *)v71 )
          {
            v4 = (unsigned __int8 *)(v71[0] + 1LL);
            j_j___libc_free_0((unsigned __int64)v69);
          }
          v26 += 4;
        }
        while ( v52 != v26 );
        if ( v25 )
        {
          v39 = v67;
          v40 = (__int64)v66;
          v41 = *(_QWORD *)(v51 + 64);
          v54 = *(_BYTE *)(v51 + 97);
          v56 = *(_BYTE *)(v51 + 96);
          v58 = *(_QWORD *)(v51 + 56);
          v42 = (_QWORD **)sub_B3B7D0(v51);
          v43 = sub_B41A60(v42, v40, v39, v58, v41, v56, v54, 0, 0);
          v44 = v43;
          if ( v43 )
          {
            v45 = sub_B3B7D0(v43);
            v46 = *(_QWORD *)(a2 - 32) == 0;
            *(_QWORD *)(a2 + 80) = v45;
            if ( !v46 )
            {
              v47 = *(_QWORD *)(a2 - 24);
              **(_QWORD **)(a2 - 16) = v47;
              if ( v47 )
                *(_QWORD *)(v47 + 16) = *(_QWORD *)(a2 - 16);
            }
            *(_QWORD *)(a2 - 32) = v44;
            v48 = *(_QWORD *)(v44 + 16);
            *(_QWORD *)(a2 - 24) = v48;
            if ( v48 )
              *(_QWORD *)(v48 + 16) = a2 - 24;
            *(_QWORD *)(a2 - 16) = v44 + 16;
            *(_QWORD *)(v44 + 16) = a2 - 32;
          }
          else
          {
            v46 = *(_QWORD *)(a2 - 32) == 0;
            *(_QWORD *)(a2 + 80) = 0;
            if ( !v46 )
            {
              v49 = *(_QWORD *)(a2 - 24);
              **(_QWORD **)(a2 - 16) = v49;
              if ( v49 )
                *(_QWORD *)(v49 + 16) = *(_QWORD *)(a2 - 16);
              *(_QWORD *)(a2 - 32) = 0;
            }
          }
          if ( v66 != v68 )
            j_j___libc_free_0((unsigned __int64)v66);
          v55 = v25;
        }
        else
        {
LABEL_85:
          if ( v66 != v68 )
            j_j___libc_free_0((unsigned __int64)v66);
        }
        goto LABEL_11;
      }
      if ( !v21 )
      {
        v22 = v68;
        goto LABEL_54;
      }
      v38 = v68;
    }
    v4 = v20;
    memcpy(v38, v20, v21);
    v21 = (size_t)v75;
    v22 = v66;
    goto LABEL_54;
  }
LABEL_11:
  if ( v63 )
    j_j___libc_free_0(v63);
  if ( v60 )
    j_j___libc_free_0(v60);
LABEL_15:
  v57 = v93;
  v10 = v93 + 192LL * v94;
  if ( v93 != v10 )
  {
    do
    {
      v11 = *(unsigned int *)(v10 - 120);
      v12 = *(_QWORD *)(v10 - 128);
      v10 -= 192LL;
      v13 = v12 + 56 * v11;
      if ( v12 != v13 )
      {
        do
        {
          v14 = *(unsigned int *)(v13 - 40);
          v15 = *(_QWORD *)(v13 - 48);
          v13 -= 56LL;
          v16 = (unsigned __int64 *)(v15 + 32 * v14);
          if ( (unsigned __int64 *)v15 != v16 )
          {
            do
            {
              v16 -= 4;
              if ( (unsigned __int64 *)*v16 != v16 + 2 )
                j_j___libc_free_0(*v16);
            }
            while ( (unsigned __int64 *)v15 != v16 );
            v15 = *(_QWORD *)(v13 + 8);
          }
          if ( v15 != v13 + 24 )
            _libc_free(v15);
        }
        while ( v12 != v13 );
        v12 = *(_QWORD *)(v10 + 64);
      }
      if ( v12 != v10 + 80 )
        _libc_free(v12);
      v17 = *(_QWORD *)(v10 + 16);
      v18 = (unsigned __int64 *)(v17 + 32LL * *(unsigned int *)(v10 + 24));
      if ( (unsigned __int64 *)v17 != v18 )
      {
        do
        {
          v18 -= 4;
          if ( (unsigned __int64 *)*v18 != v18 + 2 )
            j_j___libc_free_0(*v18);
        }
        while ( (unsigned __int64 *)v17 != v18 );
        v17 = *(_QWORD *)(v10 + 16);
      }
      if ( v17 != v10 + 32 )
        _libc_free(v17);
    }
    while ( v57 != v10 );
    v10 = v93;
  }
  if ( (char *)v10 != &v95 )
    _libc_free(v10);
  return v55;
}
