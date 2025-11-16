// Function: sub_30FFC20
// Address: 0x30ffc20
//
__int64 *__fastcall sub_30FFC20(__int64 *a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // r12
  _BYTE *v6; // rdx
  _BYTE **v8; // r13
  _BYTE **v9; // rcx
  unsigned __int64 v10; // r14
  __m128i *v11; // r15
  _BYTE **v12; // r14
  void *v13; // rcx
  _BYTE *v14; // rsi
  __int64 v15; // rbx
  unsigned __int64 v16; // rbx
  unsigned __int64 *v17; // r13
  unsigned __int64 *v18; // r15
  unsigned __int64 v19; // rdi
  void (__fastcall *v20)(_QWORD **, __int64, __int64); // rax
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // rax
  char *v24; // rax
  unsigned __int64 v25; // r14
  __int64 v26; // rax
  void *v27; // rcx
  size_t v28; // rdx
  void *v29; // rax
  __int64 v30; // rax
  __int64 v31; // [rsp+8h] [rbp-D8h]
  int v32; // [rsp+10h] [rbp-D0h]
  int v33; // [rsp+18h] [rbp-C8h]
  __int64 v34; // [rsp+20h] [rbp-C0h]
  __int64 v35; // [rsp+28h] [rbp-B8h]
  __int64 v36; // [rsp+30h] [rbp-B0h]
  __int64 v37; // [rsp+38h] [rbp-A8h]
  size_t v38; // [rsp+38h] [rbp-A8h]
  char *v40; // [rsp+48h] [rbp-98h]
  __m128i *v41; // [rsp+50h] [rbp-90h] BYREF
  __m128i *v42; // [rsp+58h] [rbp-88h]
  __m128i *v43; // [rsp+60h] [rbp-80h]
  _QWORD *v44; // [rsp+70h] [rbp-70h] BYREF
  __int64 v45; // [rsp+78h] [rbp-68h]
  _QWORD v46[2]; // [rsp+80h] [rbp-60h] BYREF
  _QWORD *v47; // [rsp+90h] [rbp-50h] BYREF
  __int64 v48; // [rsp+98h] [rbp-48h]
  _QWORD v49[8]; // [rsp+A0h] [rbp-40h] BYREF

  v4 = a1;
  v40 = a2;
  if ( !qword_5031C70 )
  {
    *a1 = 0;
    return v4;
  }
  v6 = qword_5031590;
  v41 = 0;
  v42 = 0;
  v8 = (_BYTE **)qword_5031590[1];
  v9 = (_BYTE **)qword_5031590[0];
  v43 = 0;
  v10 = qword_5031590[1] - qword_5031590[0];
  if ( qword_5031590[1] == qword_5031590[0] )
  {
    v11 = 0;
  }
  else
  {
    if ( v10 > 0x7FFFFFFFFFFFFFD0LL )
      goto LABEL_59;
    v11 = (__m128i *)sub_22077B0(qword_5031590[1] - qword_5031590[0]);
    v8 = (_BYTE **)qword_5031590[1];
    v9 = (_BYTE **)qword_5031590[0];
  }
  v41 = v11;
  v42 = v11;
  v43 = (__m128i *)((char *)v11 + v10);
  if ( v9 != v8 )
  {
    v37 = a4;
    v12 = v9;
    do
    {
      if ( v11 )
      {
        a1 = (__int64 *)v11;
        v11->m128i_i64[0] = (__int64)v11[1].m128i_i64;
        a2 = *v12;
        sub_30FA730(v11->m128i_i64, *v12, (__int64)&v12[1][(_QWORD)*v12]);
        v11[2].m128i_i32[0] = *((_DWORD *)v12 + 8);
        v11[2].m128i_i32[1] = *((_DWORD *)v12 + 9);
        v6 = v12[6];
        v16 = v6 - v12[5];
        v11[2].m128i_i64[1] = 0;
        v11[3].m128i_i64[0] = 0;
        v11[3].m128i_i64[1] = 0;
        if ( v16 )
        {
          if ( v16 > 0x7FFFFFFFFFFFFFF8LL )
            goto LABEL_59;
          v13 = (void *)sub_22077B0(v16);
        }
        else
        {
          v13 = 0;
        }
        v11[2].m128i_i64[1] = (__int64)v13;
        v11[3].m128i_i64[0] = (__int64)v13;
        v11[3].m128i_i64[1] = (__int64)v13 + v16;
        v14 = v12[5];
        v15 = v12[6] - v14;
        if ( v12[6] != v14 )
          v13 = memmove(v13, v14, v12[6] - v14);
        v11[3].m128i_i64[0] = (__int64)v13 + v15;
        v11[4].m128i_i64[0] = (__int64)v12[8];
        v11[4].m128i_i64[1] = (__int64)v12[9];
      }
      v12 += 10;
      v11 += 5;
    }
    while ( v8 != v12 );
    a4 = v37;
  }
  v42 = v11;
  if ( !(_BYTE)qword_5031B68 )
    goto LABEL_21;
  if ( v43 != v11 )
  {
    if ( !v11 )
    {
LABEL_55:
      v42 = v11 + 5;
      goto LABEL_21;
    }
    a1 = (__int64 *)v11;
    v11->m128i_i64[0] = (__int64)v11[1].m128i_i64;
    sub_30FA730(v11->m128i_i64, unk_50314E0, unk_50314E0 + unk_50314E8);
    v23 = unk_5031500;
    a2 = (char *)unk_5031508;
    v11[2].m128i_i64[1] = 0;
    v11[3].m128i_i64[0] = 0;
    v11[2].m128i_i64[0] = v23;
    v24 = (char *)unk_5031510;
    v11[3].m128i_i64[1] = 0;
    v25 = v24 - a2;
    if ( v24 == a2 )
    {
      v28 = 0;
      v25 = 0;
      v27 = 0;
      goto LABEL_52;
    }
    if ( v25 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v26 = sub_22077B0(v25);
      a2 = (char *)unk_5031508;
      v27 = (void *)v26;
      v24 = (char *)unk_5031510;
      v28 = unk_5031510 - unk_5031508;
LABEL_52:
      v11[2].m128i_i64[1] = (__int64)v27;
      v11[3].m128i_i64[0] = (__int64)v27;
      v11[3].m128i_i64[1] = (__int64)v27 + v25;
      if ( a2 != v24 )
      {
        v38 = v28;
        v29 = memmove(v27, a2, v28);
        v28 = v38;
        v27 = v29;
      }
      v30 = unk_5031520;
      v11[3].m128i_i64[0] = (__int64)v27 + v28;
      v11[4].m128i_i64[0] = v30;
      v11[4].m128i_i64[1] = unk_5031528;
      v11 = v42;
      goto LABEL_55;
    }
LABEL_59:
    sub_4261EA(a1, a2, v6);
  }
  sub_30FC6E0((unsigned __int64 *)&v41, v11, &unk_50314E0);
LABEL_21:
  v47 = v49;
  sub_30FA730((__int64 *)&v47, (_BYTE *)qword_5031C68, qword_5031C68 + qword_5031C70);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v48) <= 2
    || (sub_2241490((unsigned __int64 *)&v47, (char *)&off_3F92B2E, 3u),
        v44 = v46,
        sub_30FA730((__int64 *)&v44, (_BYTE *)qword_5031C68, qword_5031C68 + qword_5031C70),
        (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v45) <= 3) )
  {
    sub_4262D8((__int64)"basic_string::append");
  }
  sub_2241490((unsigned __int64 *)&v44, ".out", 4u);
  v32 = (int)v44;
  v33 = v45;
  v31 = *(_QWORD *)v40;
  v34 = (__int64)v47;
  v35 = v48;
  v36 = sub_22077B0(0xF8u);
  if ( v36 )
    sub_36FEDF0(v36, v31, (unsigned int)&v41, (unsigned int)&unk_5031540, v32, v33, v34, v35);
  if ( v44 != v46 )
    j_j___libc_free_0((unsigned __int64)v44);
  if ( v47 != v49 )
    j_j___libc_free_0((unsigned __int64)v47);
  v17 = (unsigned __int64 *)v42;
  v18 = (unsigned __int64 *)v41;
  if ( v42 != v41 )
  {
    do
    {
      v19 = v18[5];
      if ( v19 )
        j_j___libc_free_0(v19);
      if ( (unsigned __int64 *)*v18 != v18 + 2 )
        j_j___libc_free_0(*v18);
      v18 += 10;
    }
    while ( v17 != v18 );
    v18 = (unsigned __int64 *)v41;
  }
  if ( v18 )
    j_j___libc_free_0((unsigned __int64)v18);
  v49[0] = 0;
  v44 = (_QWORD *)v36;
  v20 = *(void (__fastcall **)(_QWORD **, __int64, __int64))(a4 + 16);
  if ( v20 )
  {
    v20(&v47, a4, 2);
    v49[1] = *(_QWORD *)(a4 + 24);
    v49[0] = *(_QWORD *)(a4 + 16);
  }
  v21 = sub_22077B0(0x178u);
  v22 = v21;
  if ( v21 )
    sub_30FED50(v21, (__int64)v40, a3, &v44, (__int64)&v47);
  if ( v49[0] )
    ((void (__fastcall *)(_QWORD **, _QWORD **, __int64))v49[0])(&v47, &v47, 3);
  if ( v44 )
    (*(void (__fastcall **)(_QWORD *))(*v44 + 8LL))(v44);
  *v4 = v22;
  return v4;
}
