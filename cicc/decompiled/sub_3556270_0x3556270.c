// Function: sub_3556270
// Address: 0x3556270
//
__int64 __fastcall sub_3556270(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r14d
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rcx
  __int64 v14; // rdx
  _QWORD *v15; // rax
  _QWORD *v16; // rdx
  unsigned __int64 v17; // rbx
  unsigned __int64 v18; // r15
  unsigned __int64 v19; // rax
  __int64 v20; // r13
  __int64 *v21; // r15
  __int64 *v22; // r13
  int v23; // edx
  signed int v24; // esi
  unsigned __int64 v25; // r12
  unsigned __int64 *v26; // rbx
  unsigned __int64 *v27; // r12
  _QWORD *v28; // rbx
  unsigned __int64 v29; // r12
  _QWORD *v31; // rdi
  __int64 v32; // rsi
  unsigned int v33; // eax
  int v34; // eax
  unsigned __int64 v35; // rax
  __int64 v36; // rax
  int v37; // ebx
  __int64 v38; // r13
  _QWORD *v39; // rax
  _QWORD *i; // rdx
  __int64 v41; // r13
  __int64 v42; // r15
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // r13
  _QWORD *v47; // r13
  unsigned __int64 v48; // r8
  unsigned __int64 v49; // r15
  unsigned int v50; // r13d
  _QWORD *v51; // r13
  _QWORD *v52; // r15
  _QWORD *v53; // rsi
  __int64 v54; // [rsp+8h] [rbp-5E8h]
  int v57; // [rsp+3Ch] [rbp-5B4h]
  unsigned __int64 v58; // [rsp+48h] [rbp-5A8h] BYREF
  __int64 v59; // [rsp+50h] [rbp-5A0h] BYREF
  char *v60; // [rsp+58h] [rbp-598h]
  __int64 v61; // [rsp+60h] [rbp-590h]
  int v62; // [rsp+68h] [rbp-588h]
  char v63; // [rsp+6Ch] [rbp-584h]
  char v64; // [rsp+70h] [rbp-580h] BYREF
  _QWORD *v65; // [rsp+90h] [rbp-560h] BYREF
  __int64 v66; // [rsp+98h] [rbp-558h]
  _QWORD *v67; // [rsp+A0h] [rbp-550h]
  __int64 v68; // [rsp+A8h] [rbp-548h]
  unsigned int v69; // [rsp+B0h] [rbp-540h]
  void **v70; // [rsp+B8h] [rbp-538h]
  int v71; // [rsp+C0h] [rbp-530h]
  void *s; // [rsp+C8h] [rbp-528h] BYREF
  unsigned int v73; // [rsp+D0h] [rbp-520h]
  char v74; // [rsp+D8h] [rbp-518h] BYREF
  _QWORD *v75; // [rsp+110h] [rbp-4E0h] BYREF
  unsigned int v76; // [rsp+118h] [rbp-4D8h]
  unsigned int v77; // [rsp+11Ch] [rbp-4D4h]
  _BYTE v78[640]; // [rsp+120h] [rbp-4D0h] BYREF
  unsigned __int64 *v79; // [rsp+3A0h] [rbp-250h]
  unsigned int v80; // [rsp+3A8h] [rbp-248h]
  char v81; // [rsp+3B0h] [rbp-240h] BYREF
  unsigned __int64 *v82; // [rsp+5B0h] [rbp-40h]
  int v83; // [rsp+5B8h] [rbp-38h]

  sub_354B460((__int64)&v65, (_QWORD *)(a1 + 48), a1 + 3528, a4, a5, a6);
  v6 = 0;
  sub_354F340((__int64)&v65, a1, v7, v8, v9, v10);
  v13 = (__int64)v78;
  v57 = (__int64)(*(_QWORD *)(a1 + 56) - *(_QWORD *)(a1 + 48)) >> 8;
  if ( v57 )
  {
    do
    {
      ++v66;
      if ( (_DWORD)v68 )
      {
        v13 = (unsigned int)(4 * v68);
        v14 = v69;
        if ( (unsigned int)v13 < 0x40 )
          v13 = 64;
        if ( v69 <= (unsigned int)v13 )
        {
LABEL_5:
          v15 = v67;
          v16 = &v67[v14];
          if ( v67 != v16 )
          {
            do
              *v15++ = -4096;
            while ( v16 != v15 );
          }
          v68 = 0;
          goto LABEL_8;
        }
        v31 = v67;
        v32 = v69;
        if ( (_DWORD)v68 == 1 )
        {
          v38 = 1024;
          v37 = 128;
        }
        else
        {
          _BitScanReverse(&v33, v68 - 1);
          v34 = 1 << (33 - (v33 ^ 0x1F));
          v13 = 64;
          if ( v34 < 64 )
            v34 = 64;
          if ( v34 == v69 )
          {
            v68 = 0;
            v53 = &v67[v32];
            do
            {
              if ( v31 )
                *v31 = -4096;
              ++v31;
            }
            while ( v53 != v31 );
            goto LABEL_8;
          }
          v35 = (4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1);
          v36 = ((((v35 >> 2) | v35 | (((v35 >> 2) | v35) >> 4)) >> 8)
               | (v35 >> 2)
               | v35
               | (((v35 >> 2) | v35) >> 4)
               | (((((v35 >> 2) | v35 | (((v35 >> 2) | v35) >> 4)) >> 8) | (v35 >> 2) | v35 | (((v35 >> 2) | v35) >> 4)) >> 16))
              + 1;
          v37 = v36;
          v38 = 8 * v36;
        }
        sub_C7D6A0((__int64)v67, v32 * 8, 8);
        v69 = v37;
        v39 = (_QWORD *)sub_C7D670(v38, 8);
        v68 = 0;
        v67 = v39;
        for ( i = &v39[v69]; i != v39; ++v39 )
        {
          if ( v39 )
            *v39 = -4096;
        }
      }
      else if ( HIDWORD(v68) )
      {
        v14 = v69;
        if ( v69 <= 0x40 )
          goto LABEL_5;
        sub_C7D6A0((__int64)v67, 8LL * v69, 8);
        v67 = 0;
        v68 = 0;
        v69 = 0;
      }
LABEL_8:
      v71 = 0;
      if ( 8LL * v73 )
        memset(s, 0, 8LL * v73);
      v63 = 1;
      v61 = 4;
      v60 = &v64;
      v62 = 0;
      v59 = 0;
      v17 = (__int64)(v65[1] - *v65) >> 8;
      v18 = v17;
      if ( v17 > v77 )
      {
        v54 = sub_C8D7D0((__int64)&v75, (__int64)v78, (__int64)(v65[1] - *v65) >> 8, 0x40u, &v58, v12);
        v46 = v54;
        do
        {
          if ( v46 )
            sub_C8CD80(v46, v46 + 32, (__int64)&v59, v43, v44, v45);
          v46 += 64;
          --v18;
        }
        while ( v18 );
        v47 = v75;
        v48 = (unsigned __int64)v76 << 6;
        v49 = (unsigned __int64)v75 + v48;
        if ( v75 != (_QWORD *)((char *)v75 + v48) )
        {
          do
          {
            while ( 1 )
            {
              v49 -= 64LL;
              if ( !*(_BYTE *)(v49 + 28) )
                break;
              if ( v47 == (_QWORD *)v49 )
                goto LABEL_76;
            }
            _libc_free(*(_QWORD *)(v49 + 8));
          }
          while ( v47 != (_QWORD *)v49 );
LABEL_76:
          v49 = (unsigned __int64)v75;
        }
        v50 = v58;
        if ( (_BYTE *)v49 != v78 )
          _libc_free(v49);
        v77 = v50;
        v76 = v17;
        v75 = (_QWORD *)v54;
      }
      else
      {
        v19 = v76;
        v20 = v76;
        if ( v17 <= v76 )
          v20 = (__int64)(v65[1] - *v65) >> 8;
        if ( v20 )
        {
          v21 = v75;
          v22 = &v75[8 * v20];
          do
          {
            if ( v21 != &v59 )
              sub_C8CE00((__int64)v21, (__int64)(v21 + 4), (__int64)&v59, v13, v11, v12);
            v21 += 8;
          }
          while ( v22 != v21 );
          v19 = v76;
        }
        if ( v17 > v19 )
        {
          v41 = (__int64)&v75[8 * v19];
          v42 = v17 - v19;
          if ( v17 != v19 )
          {
            do
            {
              if ( v41 )
                sub_C8CD80(v41, v41 + 32, (__int64)&v59, v13, v11, v12);
              v41 += 64;
              --v42;
            }
            while ( v42 );
          }
        }
        else if ( v17 < v19 )
        {
          v51 = &v75[8 * v19];
          if ( v51 != &v75[8 * v17] )
          {
            v52 = &v75[8 * v17];
            do
            {
              v51 -= 8;
              if ( !*((_BYTE *)v51 + 28) )
                _libc_free(v51[1]);
            }
            while ( v52 != v51 );
          }
        }
        v76 = v17;
      }
      if ( !v63 )
        _libc_free((unsigned __int64)v60);
      v23 = v6;
      v24 = v6;
      v83 = 0;
      ++v6;
      sub_3555D40((__int64)&v65, v24, v23, a2, a1, 0);
    }
    while ( v57 != v6 );
  }
  v25 = (unsigned __int64)v82;
  if ( v82 )
  {
    if ( *v82 )
      j_j___libc_free_0(*v82);
    j_j___libc_free_0(v25);
  }
  v26 = v79;
  v27 = &v79[4 * v80];
  if ( v79 != v27 )
  {
    do
    {
      v27 -= 4;
      if ( (unsigned __int64 *)*v27 != v27 + 2 )
        _libc_free(*v27);
    }
    while ( v26 != v27 );
    v27 = v79;
  }
  if ( v27 != (unsigned __int64 *)&v81 )
    _libc_free((unsigned __int64)v27);
  v28 = v75;
  v29 = (unsigned __int64)&v75[8 * (unsigned __int64)v76];
  if ( v75 != (_QWORD *)v29 )
  {
    do
    {
      while ( 1 )
      {
        v29 -= 64LL;
        if ( !*(_BYTE *)(v29 + 28) )
          break;
        if ( v28 == (_QWORD *)v29 )
          goto LABEL_41;
      }
      _libc_free(*(_QWORD *)(v29 + 8));
    }
    while ( v28 != (_QWORD *)v29 );
LABEL_41:
    v29 = (unsigned __int64)v75;
  }
  if ( (_BYTE *)v29 != v78 )
    _libc_free(v29);
  if ( s != &v74 )
    _libc_free((unsigned __int64)s);
  if ( v70 != &s )
    _libc_free((unsigned __int64)v70);
  return sub_C7D6A0((__int64)v67, 8LL * v69, 8);
}
