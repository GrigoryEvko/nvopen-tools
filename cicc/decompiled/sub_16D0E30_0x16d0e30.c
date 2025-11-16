// Function: sub_16D0E30
// Address: 0x16d0e30
//
__int64 __fastcall sub_16D0E30(
        __int64 a1,
        __int64 *a2,
        unsigned __int64 a3,
        int a4,
        __int64 a5,
        __int64 a6,
        unsigned __int64 *a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  int v13; // eax
  _BYTE *v14; // r8
  _QWORD *v15; // rbx
  char *(*v16)(); // rax
  _BYTE *v17; // rdx
  _BYTE *i; // r14
  char v19; // al
  _BYTE *v20; // rdx
  _BYTE **v21; // r11
  _BYTE *v22; // r8
  unsigned __int64 *v23; // rax
  __int64 v24; // r10
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // rcx
  int v27; // r9d
  __int64 v28; // rdx
  unsigned __int64 v29; // r9
  unsigned __int64 v30; // rax
  _BYTE *v31; // rdx
  __int64 v32; // rsi
  int v33; // r14^4
  int v34; // ecx
  _BYTE *v35; // rdi
  __int64 v36; // rax
  _BYTE **v37; // r11
  int v38; // r14d
  __int64 v40; // rax
  __int64 v41; // rdx
  size_t v42; // rdx
  _BYTE *v43; // rdi
  _BYTE **v44; // [rsp+8h] [rbp-108h]
  _BYTE *v45; // [rsp+10h] [rbp-100h]
  __int64 v46; // [rsp+10h] [rbp-100h]
  int v47; // [rsp+18h] [rbp-F8h]
  _BYTE *v48; // [rsp+18h] [rbp-F8h]
  _BYTE *v50; // [rsp+20h] [rbp-F0h]
  _QWORD *v51; // [rsp+20h] [rbp-F0h]
  unsigned __int64 *v52; // [rsp+20h] [rbp-F0h]
  __int64 v53; // [rsp+28h] [rbp-E8h]
  int v55; // [rsp+38h] [rbp-D8h]
  _BYTE **v57; // [rsp+40h] [rbp-D0h]
  unsigned __int64 v58; // [rsp+40h] [rbp-D0h]
  __int64 v59; // [rsp+50h] [rbp-C0h]
  char *v60; // [rsp+58h] [rbp-B8h]
  _QWORD *dest; // [rsp+70h] [rbp-A0h]
  size_t v62; // [rsp+78h] [rbp-98h]
  _QWORD v63[2]; // [rsp+80h] [rbp-90h] BYREF
  _BYTE *v64; // [rsp+90h] [rbp-80h] BYREF
  size_t n; // [rsp+98h] [rbp-78h]
  _QWORD src[2]; // [rsp+A0h] [rbp-70h] BYREF
  _BYTE *v67; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v68; // [rsp+B8h] [rbp-58h]
  _BYTE v69[80]; // [rsp+C0h] [rbp-50h] BYREF

  v67 = v69;
  v68 = 0x400000000LL;
  v60 = "<unknown>";
  v59 = 9;
  dest = v63;
  LOBYTE(v63[0]) = 0;
  if ( a3 )
  {
    v13 = sub_16CE270(a2, a3);
    v14 = (_BYTE *)a3;
    v55 = v13;
    v15 = *(_QWORD **)(*a2 + 24LL * (unsigned int)(v13 - 1));
    v16 = *(char *(**)())(*v15 + 16LL);
    if ( v16 == sub_12BCB10 )
    {
      v59 = 14;
      v60 = "Unknown buffer";
    }
    else
    {
      v40 = ((__int64 (__fastcall *)(_QWORD *))v16)(v15);
      v14 = (_BYTE *)a3;
      v60 = (char *)v40;
      v59 = v41;
    }
    v17 = (_BYTE *)v15[1];
    for ( i = (_BYTE *)a3; i != v17; --i )
    {
      v19 = *(i - 1);
      if ( v19 == 10 )
        break;
      if ( v19 == 13 )
        break;
    }
    v20 = (_BYTE *)v15[2];
    if ( (_BYTE *)a3 != v20 )
    {
      do
      {
        if ( *v14 == 13 )
          break;
        if ( *v14 == 10 )
          break;
        ++v14;
      }
      while ( v14 != v20 );
    }
    v45 = v14;
    v64 = src;
    sub_16CD2C0((__int64 *)&v64, i, (__int64)v14);
    v21 = &v64;
    v22 = v45;
    if ( v64 == (_BYTE *)src )
    {
      v42 = n;
      if ( n )
      {
        if ( n == 1 )
        {
          LOBYTE(v63[0]) = src[0];
          v42 = 1;
        }
        else
        {
          memcpy(v63, src, n);
          v42 = n;
          v21 = &v64;
          v22 = v45;
        }
      }
      v62 = v42;
      *((_BYTE *)v63 + v42) = 0;
      v43 = v64;
    }
    else
    {
      dest = v64;
      v62 = n;
      v63[0] = src[0];
      v64 = src;
      v43 = src;
    }
    n = 0;
    *v43 = 0;
    if ( v64 != (_BYTE *)src )
    {
      v50 = v22;
      j_j___libc_free_0(v64, src[0] + 1LL);
      v21 = &v64;
      v22 = v50;
    }
    if ( (_DWORD)a8 )
    {
      v23 = a7;
      v24 = (__int64)&a7[2 * (unsigned int)(a8 - 1) + 2];
      do
      {
        while ( 1 )
        {
          v25 = *v23;
          v26 = v23[1];
          if ( *v23 != 0 && *v23 <= (unsigned __int64)v22 && v26 >= (unsigned __int64)i )
            break;
          v23 += 2;
          if ( v23 == (unsigned __int64 *)v24 )
            goto LABEL_30;
        }
        if ( v25 < (unsigned __int64)i )
          LODWORD(v25) = (_DWORD)i;
        if ( v26 > (unsigned __int64)v22 )
          LODWORD(v26) = (_DWORD)v22;
        v27 = v25;
        v28 = (unsigned int)v68;
        v29 = ((unsigned __int64)(unsigned int)(v26 - (_DWORD)i) << 32) | (unsigned int)(v27 - (_DWORD)i);
        if ( (unsigned int)v68 >= HIDWORD(v68) )
        {
          v44 = v21;
          v46 = v24;
          v48 = v22;
          v52 = v23;
          v58 = v29;
          sub_16CD150((__int64)&v67, v69, 0, 8, (int)v22, v29);
          v28 = (unsigned int)v68;
          v21 = v44;
          v24 = v46;
          v22 = v48;
          v23 = v52;
          v29 = v58;
        }
        v23 += 2;
        *(_QWORD *)&v67[8 * v28] = v29;
        LODWORD(v68) = v68 + 1;
      }
      while ( v23 != (unsigned __int64 *)v24 );
    }
LABEL_30:
    v57 = v21;
    v30 = sub_16CFA40(a2, a3, v55);
    v31 = v67;
    v32 = v62;
    v33 = HIDWORD(v30);
    v34 = v30;
    v35 = dest;
    v36 = (unsigned int)v68;
    v37 = v57;
    v38 = v33 - 1;
  }
  else
  {
    v35 = v63;
    v32 = 0;
    v36 = 0;
    v34 = 0;
    v31 = v69;
    v38 = -1;
    v37 = &v64;
  }
  v47 = v34;
  v51 = v31;
  v53 = v36;
  sub_16E2FC0(v37, a5);
  sub_16D0AA0(a1, (__int64)a2, a3, v60, v59, v47, v38, a4, v64, n, v35, v32, v51, v53, a9, a10);
  if ( v64 != (_BYTE *)src )
    j_j___libc_free_0(v64, src[0] + 1LL);
  if ( dest != v63 )
    j_j___libc_free_0(dest, v63[0] + 1LL);
  if ( v67 != v69 )
    _libc_free((unsigned __int64)v67);
  return a1;
}
