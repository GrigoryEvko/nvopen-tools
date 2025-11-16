// Function: sub_B28710
// Address: 0xb28710
//
_QWORD *__fastcall sub_B28710(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rax
  int v8; // ebx
  __int64 v9; // r13
  __int64 *v10; // r14
  int v11; // eax
  unsigned int v12; // eax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 *v17; // r12
  __int64 v18; // r15
  __int64 i; // rbx
  __int64 v20; // rsi
  _QWORD *v21; // rax
  unsigned int v22; // eax
  unsigned int v23; // ebx
  int v24; // ecx
  __int64 v25; // rax
  unsigned int v26; // r11d
  __int64 *v27; // r12
  int v28; // r14d
  __int64 v29; // rax
  char *v30; // rbx
  char *v31; // r14
  char *v32; // rdi
  _QWORD *v33; // r10
  unsigned int v34; // r8d
  __int64 *v35; // r14
  int v36; // r8d
  unsigned int v37; // r9d
  __int64 v38; // rdx
  __int64 *v39; // rax
  unsigned int v40; // r11d
  _BYTE *v41; // rbx
  _BYTE *v42; // r12
  _BYTE *v43; // rdi
  __int64 v46; // [rsp+18h] [rbp-2148h]
  int v49; // [rsp+38h] [rbp-2128h]
  __int64 *v50; // [rsp+38h] [rbp-2128h]
  __int64 v51; // [rsp+40h] [rbp-2120h]
  __int64 v52; // [rsp+40h] [rbp-2120h]
  unsigned int v53; // [rsp+40h] [rbp-2120h]
  __int64 v54; // [rsp+40h] [rbp-2120h]
  unsigned int v55; // [rsp+58h] [rbp-2108h]
  __int64 v56[4]; // [rsp+60h] [rbp-2100h] BYREF
  __int128 v57; // [rsp+80h] [rbp-20E0h] BYREF
  __int128 v58; // [rsp+90h] [rbp-20D0h] BYREF
  __int64 v59; // [rsp+A0h] [rbp-20C0h]
  _QWORD *v60; // [rsp+D0h] [rbp-2090h] BYREF
  __int64 v61; // [rsp+D8h] [rbp-2088h]
  _QWORD v62[64]; // [rsp+E0h] [rbp-2080h] BYREF
  _BYTE *v63; // [rsp+2E0h] [rbp-1E80h]
  __int64 v64; // [rsp+2E8h] [rbp-1E78h]
  _BYTE v65[3584]; // [rsp+2F0h] [rbp-1E70h] BYREF
  __int64 v66; // [rsp+10F0h] [rbp-1070h]
  __int64 *v67; // [rsp+1100h] [rbp-1060h] BYREF
  __int64 v68; // [rsp+1108h] [rbp-1058h]
  __int64 v69; // [rsp+1110h] [rbp-1050h] BYREF
  char *v70[2]; // [rsp+1118h] [rbp-1048h] BYREF
  _BYTE v71[488]; // [rsp+1128h] [rbp-1038h] BYREF
  char *v72; // [rsp+1310h] [rbp-E50h]
  __int64 v73; // [rsp+1318h] [rbp-E48h]
  char v74; // [rsp+1320h] [rbp-E40h] BYREF
  __int64 v75; // [rsp+2120h] [rbp-40h]

  *a1 = a1 + 2;
  a1[1] = 0x400000000LL;
  v60 = v62;
  v61 = 0x4000000001LL;
  v63 = v65;
  v66 = a3;
  v62[0] = 0;
  v64 = 0x4000000000LL;
  v5 = sub_B20CA0((__int64)&v60, 0);
  v6 = 0;
  *(_QWORD *)(v5 + 8) = 0x100000001LL;
  *(_DWORD *)v5 = 1;
  sub_B1A4E0((__int64)&v60, 0);
  v7 = *(_QWORD *)(a2 + 128);
  v51 = v7 + 72;
  if ( *(_QWORD *)(v7 + 80) == v7 + 72 )
    goto LABEL_51;
  v8 = 0;
  v9 = *(_QWORD *)(v7 + 80);
  v55 = 1;
  while ( 1 )
  {
    v10 = 0;
    if ( v9 )
      v10 = (__int64 *)(v9 - 24);
    v6 = (__int64)v10;
    sub_B1CB80(&v67, (__int64)v10, a3);
    v11 = v68;
    if ( v67 != &v69 )
    {
      v49 = v68;
      _libc_free(v67, v10);
      v11 = v49;
    }
    if ( !v11 )
      break;
    v9 = *(_QWORD *)(v9 + 8);
    if ( v51 == v9 )
      goto LABEL_11;
LABEL_4:
    ++v8;
  }
  sub_B1A4E0((__int64)a1, (__int64)v10);
  v6 = (__int64)v10;
  v12 = sub_B27790((__int64)&v60, v10, v55, (unsigned __int8 (__fastcall *)(__int64, __int64))sub_B184A0, 1u, 0);
  v9 = *(_QWORD *)(v9 + 8);
  v55 = v12;
  if ( v51 != v9 )
    goto LABEL_4;
LABEL_11:
  if ( v55 != v8 + 2 )
  {
    v59 = 0;
    v57 = 0;
    v56[1] = a2;
    v13 = *(_QWORD *)(a2 + 128);
    v58 = 0;
    v14 = v13 + 72;
    v15 = *(_QWORD *)(v13 + 80);
    v56[0] = (__int64)&v57;
    v56[2] = (__int64)&v60;
    v52 = v14;
    if ( v14 != v15 )
    {
      v16 = v15;
      do
      {
        while ( 1 )
        {
          v17 = (__int64 *)(v16 - 24);
          if ( !v16 )
            v17 = 0;
          v6 = *(unsigned int *)sub_B20CA0((__int64)&v60, (__int64)v17);
          if ( !(_DWORD)v6 )
            break;
          v16 = *(_QWORD *)(v16 + 8);
          if ( v52 == v16 )
            goto LABEL_28;
        }
        if ( !(_BYTE)v59 )
          sub_B23260(v56);
        v18 = (unsigned int)sub_B282E0(
                              (__int64)&v60,
                              v17,
                              v55,
                              (unsigned __int8 (__fastcall *)(__int64, __int64))sub_B184A0,
                              v55,
                              (__int64 *)&v57);
        v50 = (__int64 *)v60[v18];
        sub_B1A4E0((__int64)a1, (__int64)v50);
        if ( (unsigned int)v18 > v55 )
        {
          v46 = v16;
          for ( i = v18; ; --i )
          {
            v20 = v60[i];
            v70[0] = v71;
            v67 = 0;
            v68 = 0;
            v69 = 0;
            v70[1] = (char *)0x400000000LL;
            v21 = (_QWORD *)sub_B20CA0((__int64)&v60, v20);
            *v21 = v67;
            v21[1] = v68;
            v21[2] = v69;
            sub_B189E0((__int64)(v21 + 3), v70);
            if ( v70[0] != v71 )
              _libc_free(v70[0], v70);
            LODWORD(v61) = v61 - 1;
            if ( v18 - ((unsigned int)v18 + ~v55) == i )
              break;
          }
          v16 = v46;
        }
        v6 = (__int64)v50;
        v22 = sub_B27790((__int64)&v60, v50, v55, (unsigned __int8 (__fastcall *)(__int64, __int64))sub_B184A0, 1u, 0);
        v16 = *(_QWORD *)(v16 + 8);
        v55 = v22;
      }
      while ( v52 != v16 );
LABEL_28:
      if ( (_BYTE)v59 )
      {
        LOBYTE(v59) = 0;
        v6 = 16LL * DWORD2(v58);
        sub_C7D6A0(*((_QWORD *)&v57 + 1), v6, 8);
      }
    }
    v23 = 0;
    v67 = &v69;
    v68 = 0x4000000001LL;
    v24 = *((_DWORD *)a1 + 2);
    v72 = &v74;
    v73 = 0x4000000000LL;
    v69 = 0;
    v75 = a3;
    v25 = 0;
    if ( v24 )
    {
      do
      {
        while ( 1 )
        {
          v27 = (__int64 *)(*a1 + 8 * v25);
          v6 = *v27;
          sub_B1CB80(&v57, *v27, a3);
          v28 = DWORD2(v57);
          if ( (__int128 *)v57 != &v58 )
            _libc_free(v57, v6);
          if ( !v28 )
            break;
          v29 = 0;
          LODWORD(v68) = 0;
          if ( !HIDWORD(v68) )
          {
            v6 = (__int64)&v69;
            sub_C8D5F0(&v67, &v69, 1, 8);
            v29 = (unsigned int)v68;
          }
          v67[v29] = 0;
          LODWORD(v68) = v68 + 1;
          if ( v72 != &v72[56 * (unsigned int)v73] )
          {
            v53 = v23;
            v30 = &v72[56 * (unsigned int)v73];
            v31 = v72;
            do
            {
              v30 -= 56;
              v32 = (char *)*((_QWORD *)v30 + 3);
              if ( v32 != v30 + 40 )
                _libc_free(v32, v6);
            }
            while ( v31 != v30 );
            v23 = v53;
          }
          LODWORD(v73) = 0;
          v6 = *v27;
          if ( (unsigned int)sub_B282E0(
                               (__int64)&v67,
                               (__int64 *)*v27,
                               0,
                               (unsigned __int8 (__fastcall *)(__int64, __int64))sub_B184A0,
                               0,
                               0) <= 1 )
            break;
          v33 = (_QWORD *)*a1;
          v34 = 2;
          v35 = v67;
          v54 = *((unsigned int *)a1 + 2);
          v6 = *a1 + v54 * 8;
          while ( 1 )
          {
            *(_QWORD *)&v57 = v35[v34];
            if ( (_QWORD *)v6 != sub_B18540(v33, v6, (__int64 *)&v57) )
              break;
            v34 = v36 + 1;
            if ( v37 < v34 )
              goto LABEL_33;
          }
          v38 = *v27;
          v39 = &v33[v54 - 1];
          *v27 = *v39;
          *v39 = v38;
          v40 = *((_DWORD *)a1 + 2) - 1;
          v25 = v23;
          *((_DWORD *)a1 + 2) = v40;
          if ( v40 <= v23 )
            goto LABEL_50;
        }
        v26 = *((_DWORD *)a1 + 2);
LABEL_33:
        v25 = ++v23;
      }
      while ( v26 > v23 );
    }
LABEL_50:
    sub_B1ACF0((__int64)&v67, v6);
  }
LABEL_51:
  v41 = v63;
  v42 = &v63[56 * (unsigned int)v64];
  if ( v63 != v42 )
  {
    do
    {
      v42 -= 56;
      v43 = (_BYTE *)*((_QWORD *)v42 + 3);
      if ( v43 != v42 + 40 )
        _libc_free(v43, v6);
    }
    while ( v41 != v42 );
    v42 = v63;
  }
  if ( v42 != v65 )
    _libc_free(v42, v6);
  if ( v60 != v62 )
    _libc_free(v60, v6);
  return a1;
}
