// Function: sub_239BE70
// Address: 0x239be70
//
_QWORD *__fastcall sub_239BE70(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  _QWORD *v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  _QWORD *v12; // rbx
  __int64 v13; // rdx
  _BYTE *v14; // r14
  char *v15; // rbx
  char *v16; // r12
  unsigned __int64 v17; // r14
  unsigned __int64 v18; // rdi
  _BYTE *v20; // r12
  unsigned __int64 v21; // r15
  unsigned __int64 v22; // rdi
  _BYTE *v23; // r12
  unsigned __int64 v24; // r15
  unsigned __int64 v25; // rdi
  char *v26; // rbx
  char *v27; // r14
  unsigned __int64 v28; // r15
  unsigned __int64 v29; // rdi
  char *v30; // [rsp+10h] [rbp-170h] BYREF
  int v31; // [rsp+18h] [rbp-168h]
  char v32; // [rsp+20h] [rbp-160h] BYREF
  char *v33; // [rsp+40h] [rbp-140h] BYREF
  int v34; // [rsp+48h] [rbp-138h]
  char v35; // [rsp+50h] [rbp-130h] BYREF
  __int64 v36; // [rsp+88h] [rbp-F8h]
  __int64 v37; // [rsp+90h] [rbp-F0h]
  char v38; // [rsp+98h] [rbp-E8h]
  __int64 v39; // [rsp+9Ch] [rbp-E4h]
  char *v40; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v41; // [rsp+B8h] [rbp-C8h]
  _BYTE v42[32]; // [rsp+C0h] [rbp-C0h] BYREF
  _BYTE *v43; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v44; // [rsp+E8h] [rbp-98h]
  _BYTE v45[56]; // [rsp+F0h] [rbp-90h] BYREF
  __int64 v46; // [rsp+128h] [rbp-58h]
  __int64 v47; // [rsp+130h] [rbp-50h]
  char v48; // [rsp+138h] [rbp-48h]
  __int64 v49; // [rsp+13Ch] [rbp-44h]

  sub_104C770((__int64)&v30, a2 + 8, a3);
  v40 = v42;
  v41 = 0x400000000LL;
  if ( v31 )
    sub_2303CE0((__int64)&v40, &v30, v4, v5, v6, v7);
  v43 = v45;
  v44 = 0x600000000LL;
  if ( v34 )
  {
    sub_239BAB0((__int64)&v43, (__int64)&v33);
    v26 = v33;
    v46 = v36;
    v47 = v37;
    v48 = v38;
    v49 = v39;
    v27 = &v33[8 * v34];
    while ( v26 != v27 )
    {
      while ( 1 )
      {
        v28 = *((_QWORD *)v27 - 1);
        v27 -= 8;
        if ( !v28 )
          break;
        v29 = *(_QWORD *)(v28 + 24);
        if ( v29 != v28 + 40 )
          _libc_free(v29);
        j_j___libc_free_0(v28);
        if ( v26 == v27 )
          goto LABEL_5;
      }
    }
  }
  else
  {
    v46 = v36;
    v47 = v37;
    v48 = v38;
    v49 = v39;
  }
LABEL_5:
  v34 = 0;
  v36 = 0;
  v37 = 0;
  v8 = (_QWORD *)sub_22077B0(0xA0u);
  v12 = v8;
  if ( v8 )
  {
    v13 = (unsigned int)v41;
    *v8 = &unk_4A15958;
    v8[1] = v8 + 3;
    v8[2] = 0x400000000LL;
    if ( (_DWORD)v13 )
      sub_2303CE0((__int64)(v8 + 1), &v40, v13, v9, v10, v11);
    v12[7] = v12 + 9;
    v12[8] = 0x600000000LL;
    if ( (_DWORD)v44 )
    {
      sub_239BAB0((__int64)(v12 + 7), (__int64)&v43);
      v20 = v43;
      v12[16] = v46;
      v12[17] = v47;
      *((_BYTE *)v12 + 144) = v48;
      *(_QWORD *)((char *)v12 + 148) = v49;
      v14 = &v20[8 * (unsigned int)v44];
      if ( v20 != v14 )
      {
        do
        {
          v21 = *((_QWORD *)v14 - 1);
          v14 -= 8;
          if ( v21 )
          {
            v22 = *(_QWORD *)(v21 + 24);
            if ( v22 != v21 + 40 )
              _libc_free(v22);
            j_j___libc_free_0(v21);
          }
        }
        while ( v20 != v14 );
        v14 = v43;
      }
    }
    else
    {
      v14 = v43;
      v12[16] = v46;
      v12[17] = v47;
      *((_BYTE *)v12 + 144) = v48;
      *(_QWORD *)((char *)v12 + 148) = v49;
    }
  }
  else
  {
    v23 = v43;
    v14 = &v43[8 * (unsigned int)v44];
    if ( v43 != v14 )
    {
      do
      {
        v24 = *((_QWORD *)v14 - 1);
        v14 -= 8;
        if ( v24 )
        {
          v25 = *(_QWORD *)(v24 + 24);
          if ( v25 != v24 + 40 )
            _libc_free(v25);
          j_j___libc_free_0(v24);
        }
      }
      while ( v23 != v14 );
      v14 = v43;
    }
  }
  if ( v14 != v45 )
    _libc_free((unsigned __int64)v14);
  if ( v40 != v42 )
    _libc_free((unsigned __int64)v40);
  *a1 = v12;
  v15 = v33;
  v16 = &v33[8 * v34];
  if ( v33 != v16 )
  {
    do
    {
      v17 = *((_QWORD *)v16 - 1);
      v16 -= 8;
      if ( v17 )
      {
        v18 = *(_QWORD *)(v17 + 24);
        if ( v18 != v17 + 40 )
          _libc_free(v18);
        j_j___libc_free_0(v17);
      }
    }
    while ( v15 != v16 );
    v16 = v33;
  }
  if ( v16 != &v35 )
    _libc_free((unsigned __int64)v16);
  if ( v30 != &v32 )
    _libc_free((unsigned __int64)v30);
  return a1;
}
