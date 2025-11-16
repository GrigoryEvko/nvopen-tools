// Function: sub_1696580
// Address: 0x1696580
//
__int64 *__fastcall sub_1696580(__int64 *a1, _QWORD *a2, __int64 a3, char a4, __int64 a5)
{
  __int64 v5; // rdx
  _QWORD *v6; // r13
  _QWORD *v8; // r15
  __int64 v9; // rcx
  __int64 v10; // rcx
  __int64 v11; // rcx
  unsigned __int64 v12; // r14
  _QWORD *v13; // r9
  _BYTE *v14; // rcx
  unsigned __int64 v15; // rdx
  char i; // al
  __int64 v17; // rcx
  unsigned __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v21; // rdx
  _BYTE *v22; // rax
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // r14
  unsigned __int64 v25; // r8
  char v26; // si
  unsigned __int64 j; // rdi
  unsigned __int64 v28; // rdx
  unsigned __int64 v29; // rcx
  _QWORD *v32; // [rsp+18h] [rbp-128h]
  _BYTE *v33; // [rsp+18h] [rbp-128h]
  unsigned __int64 v34; // [rsp+18h] [rbp-128h]
  _QWORD v35[3]; // [rsp+28h] [rbp-118h] BYREF
  unsigned __int64 v36; // [rsp+40h] [rbp-100h] BYREF
  unsigned __int64 v37; // [rsp+48h] [rbp-F8h] BYREF
  _BYTE v38[16]; // [rsp+50h] [rbp-F0h] BYREF
  _QWORD *v39; // [rsp+60h] [rbp-E0h] BYREF
  unsigned __int64 v40; // [rsp+68h] [rbp-D8h]
  _QWORD v41[2]; // [rsp+70h] [rbp-D0h] BYREF
  _BYTE *v42; // [rsp+80h] [rbp-C0h]
  __int64 v43; // [rsp+88h] [rbp-B8h]
  _BYTE v44[176]; // [rsp+90h] [rbp-B0h] BYREF

  v5 = 4 * a3;
  v6 = &a2[v5];
  v39 = v41;
  v40 = 0;
  LOBYTE(v41[0]) = 0;
  if ( &a2[v5] == a2 )
  {
    v13 = v41;
    v12 = 0;
    i = 0;
    v14 = v38;
  }
  else
  {
    v8 = a2 + 4;
    sub_2240E30(&v39, a2[1] + ((v5 * 8) >> 5) - 1 + a2[1] * ((unsigned __int64)(v5 * 8 - 32) >> 5));
    sub_2241490(&v39, *a2, a2[1], v9);
    if ( a2 + 4 != v6 )
    {
      while ( v40 != 0x3FFFFFFFFFFFFFFFLL )
      {
        v8 += 4;
        sub_2241490(&v39, &unk_3F871B2, 1, v10);
        sub_2241490(&v39, *(v8 - 4), *(v8 - 3), v11);
        if ( v6 == v8 )
          goto LABEL_5;
      }
LABEL_32:
      sub_4262D8((__int64)"basic_string::append");
    }
LABEL_5:
    v12 = v40;
    v13 = v39;
    v14 = v38;
    v15 = v40 >> 7;
    for ( i = v40 & 0x7F; v15; v15 >>= 7 )
    {
      *v14++ = i | 0x80;
      i = v15 & 0x7F;
    }
  }
  *v14 = i;
  v17 = (_DWORD)v14 + 1 - (unsigned int)v38;
  if ( a4 )
  {
    v33 = &v38[v17];
    v43 = 0x8000000000LL;
    v42 = v44;
    def_16BF26D();
    v21 = v35[0];
    v22 = v33;
    v35[0] = 0;
    v23 = v21 & 0xFFFFFFFFFFFFFFFELL;
    if ( v23 )
    {
      v35[1] = 0;
      v37 = v23 | 1;
      v35[2] = 0;
      sub_16963B0((__int64 *)&v36, (__int64 *)&v37);
      if ( (v36 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v36 = v36 & 0xFFFFFFFFFFFFFFFELL | 1;
        sub_16BCAE0(&v36);
      }
      if ( (v37 & 1) != 0 || (v37 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_16BCAE0(&v37);
      sub_1693CB0(a1, 15);
    }
    else
    {
      v24 = (unsigned int)v43;
      v25 = (unsigned __int64)v42;
      v26 = v43 & 0x7F;
      for ( j = (unsigned __int64)(unsigned int)v43 >> 7; j; j >>= 7 )
      {
        *v22++ = v26 | 0x80;
        v26 = j & 0x7F;
      }
      *v22 = v26;
      v28 = (_DWORD)v22 + 1 - (unsigned int)v38;
      if ( v28 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a5 + 8) )
        goto LABEL_32;
      v34 = v25;
      sub_2241490(a5, v38, v28, 0x3FFFFFFFFFFFFFFFLL);
      v29 = 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a5 + 8);
      if ( v24 > v29 )
        goto LABEL_32;
      sub_2241490(a5, v34, v24, v29);
      *a1 = 1;
    }
    if ( (v35[0] & 1) != 0 || (v35[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_16BCAE0(v35);
    if ( v42 != v44 )
      _libc_free((unsigned __int64)v42);
  }
  else
  {
    v38[v17] = 0;
    v18 = (unsigned int)(v17 + 1);
    v32 = v13;
    if ( v18 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a5 + 8) )
      goto LABEL_32;
    sub_2241490(a5, v38, v18, v17);
    if ( 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a5 + 8) < v12 )
      goto LABEL_32;
    sub_2241490(a5, v32, v12, v19);
    *a1 = 1;
  }
  if ( v39 != v41 )
    j_j___libc_free_0(v39, v41[0] + 1LL);
  return a1;
}
