// Function: sub_C917B0
// Address: 0xc917b0
//
__int64 __fastcall sub_C917B0(
        __int64 a1,
        __int64 *a2,
        unsigned __int64 a3,
        int a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        const __m128i *a9,
        __int64 a10)
{
  int v12; // eax
  _BYTE *v13; // r8
  int v14; // r10d
  __int64 v15; // r11
  __int64 v16; // rcx
  _QWORD *v17; // rsi
  char *(*v18)(); // rax
  __int64 v19; // rdx
  unsigned __int64 i; // rbx
  char v21; // al
  _BYTE *v22; // rdx
  unsigned __int64 v23; // r9
  unsigned __int64 *v24; // r11
  unsigned __int64 *v25; // rax
  unsigned __int64 v26; // rcx
  unsigned __int64 v27; // rbx
  __int64 v28; // rdx
  unsigned __int64 v29; // rbx
  unsigned __int64 v30; // rax
  _BYTE *v31; // rsi
  __int64 v32; // rcx
  int v33; // r9d
  int v34; // ebx
  __int64 *v35; // rsi
  __int64 v37; // rax
  __int64 v38; // rdx
  unsigned __int64 *v39; // [rsp+0h] [rbp-E0h]
  unsigned __int64 *v40; // [rsp+8h] [rbp-D8h]
  _BYTE *v41; // [rsp+10h] [rbp-D0h]
  int v42; // [rsp+18h] [rbp-C8h]
  int v43; // [rsp+18h] [rbp-C8h]
  unsigned __int64 v44; // [rsp+18h] [rbp-C8h]
  _QWORD *v46; // [rsp+20h] [rbp-C0h]
  int v47; // [rsp+20h] [rbp-C0h]
  __int64 v48; // [rsp+28h] [rbp-B8h]
  __int64 v51; // [rsp+40h] [rbp-A0h]
  char *v52; // [rsp+48h] [rbp-98h]
  _BYTE *v53; // [rsp+50h] [rbp-90h]
  __int64 v54; // [rsp+58h] [rbp-88h]
  _BYTE *v55[2]; // [rsp+60h] [rbp-80h] BYREF
  __int64 v56; // [rsp+70h] [rbp-70h] BYREF
  _BYTE *v57; // [rsp+80h] [rbp-60h] BYREF
  __int64 v58; // [rsp+88h] [rbp-58h]
  _BYTE v59[80]; // [rsp+90h] [rbp-50h] BYREF

  v58 = 0x400000000LL;
  v57 = v59;
  v52 = "<unknown>";
  v51 = 9;
  v53 = 0;
  v54 = 0;
  if ( a3 )
  {
    v12 = sub_C8ED90(a2, a3);
    v13 = (_BYTE *)a3;
    v14 = v12;
    v15 = a8;
    v16 = a7;
    v17 = *(_QWORD **)(*a2 + 24LL * (unsigned int)(v12 - 1));
    v18 = *(char *(**)())(*v17 + 16LL);
    if ( v18 == sub_C1E8B0 )
    {
      v51 = 14;
      v52 = "Unknown buffer";
    }
    else
    {
      v43 = v14;
      v37 = ((__int64 (__fastcall *)(_QWORD *))v18)(v17);
      v15 = a8;
      v16 = a7;
      v52 = (char *)v37;
      v13 = (_BYTE *)a3;
      v51 = v38;
      v14 = v43;
    }
    v19 = v17[1];
    for ( i = a3; i != v19; --i )
    {
      v21 = *(_BYTE *)(i - 1);
      if ( v21 == 13 )
        break;
      if ( v21 == 10 )
        break;
    }
    v22 = (_BYTE *)v17[2];
    if ( (_BYTE *)a3 != v22 )
    {
      do
      {
        if ( *v13 == 13 )
          break;
        if ( *v13 == 10 )
          break;
        ++v13;
      }
      while ( v22 != v13 );
    }
    v53 = (_BYTE *)i;
    v23 = i;
    v24 = (unsigned __int64 *)(v16 + 16 * v15);
    v54 = (__int64)&v13[-i];
    v25 = (unsigned __int64 *)v16;
    if ( (unsigned __int64 *)v16 != v24 )
    {
      do
      {
        while ( 1 )
        {
          v26 = v25[1];
          if ( *v25 <= (unsigned __int64)v13 && *v25 != 0 && v26 >= v23 )
            break;
          v25 += 2;
          if ( v24 == v25 )
            goto LABEL_26;
        }
        v27 = *v25;
        v28 = (unsigned int)v58;
        if ( v23 >= *v25 )
          LODWORD(v27) = v23;
        if ( v26 > (unsigned __int64)v13 )
          LODWORD(v26) = (_DWORD)v13;
        v29 = ((unsigned __int64)(unsigned int)(v26 - v23) << 32) | (unsigned int)(v27 - v23);
        if ( (unsigned __int64)(unsigned int)v58 + 1 > HIDWORD(v58) )
        {
          v39 = v24;
          v40 = v25;
          v41 = v13;
          v44 = v23;
          v47 = v14;
          sub_C8D5F0((__int64)&v57, v59, (unsigned int)v58 + 1LL, 8u, (__int64)v13, v23);
          v28 = (unsigned int)v58;
          v24 = v39;
          v25 = v40;
          v13 = v41;
          v23 = v44;
          v14 = v47;
        }
        v25 += 2;
        *(_QWORD *)&v57[8 * v28] = v29;
        LODWORD(v58) = v58 + 1;
      }
      while ( v24 != v25 );
    }
LABEL_26:
    v30 = sub_C90410(a2, a3, v14);
    v31 = v57;
    v32 = (unsigned int)v58;
    v33 = v30;
    v34 = HIDWORD(v30) - 1;
  }
  else
  {
    v32 = 0;
    v31 = v59;
    v33 = 0;
    v34 = -1;
  }
  v46 = v31;
  v42 = v33;
  v48 = v32;
  sub_CA0F50(v55, a5);
  v35 = a2;
  sub_C91410(a1, (__int64)a2, a3, v52, v51, v42, v34, a4, v55[0], (__int64)v55[1], v53, v54, v46, v48, a9, a10);
  if ( (__int64 *)v55[0] != &v56 )
  {
    v35 = (__int64 *)(v56 + 1);
    j_j___libc_free_0(v55[0], v56 + 1);
  }
  if ( v57 != v59 )
    _libc_free(v57, v35);
  return a1;
}
