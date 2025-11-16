// Function: sub_1274FF0
// Address: 0x1274ff0
//
__int64 __fastcall sub_1274FF0(const char *src)
{
  void *v2; // rax
  size_t v3; // r14
  _QWORD *v4; // rdx
  unsigned int v5; // r12d
  unsigned __int64 v6; // rax
  __int64 **v7; // rax
  __m128i *v9; // rax
  __int64 v10; // rsi
  __int64 *v11; // rdi
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // r14
  void *v15; // rax
  __int64 *v16; // rcx
  unsigned __int64 v17; // r14
  __int64 **v18; // rax
  _QWORD *v19; // rdi
  __int64 v20; // rax
  _QWORD *v21; // r15
  _BYTE *v22; // r9
  size_t v23; // r8
  _BYTE *v24; // rdi
  char v25; // al
  __int64 *v26; // r9
  unsigned __int64 v27; // r8
  __int64 v28; // r14
  __int64 *v29; // rax
  _QWORD *v30; // rdx
  __int64 v31; // rax
  size_t v32; // r14
  void *v33; // rax
  __int64 *v34; // rax
  _QWORD *v35; // rsi
  unsigned __int64 v36; // rdi
  _QWORD *v37; // rcx
  unsigned __int64 v38; // rdx
  __int64 *v39; // rax
  __int64 v40; // rdx
  size_t v41; // [rsp+8h] [rbp-88h]
  unsigned __int64 v42; // [rsp+8h] [rbp-88h]
  _BYTE *v43; // [rsp+10h] [rbp-80h]
  unsigned __int64 v44; // [rsp+10h] [rbp-80h]
  __int64 *v45; // [rsp+10h] [rbp-80h]
  size_t n; // [rsp+18h] [rbp-78h]
  void *srca; // [rsp+20h] [rbp-70h] BYREF
  size_t v48; // [rsp+28h] [rbp-68h]
  _QWORD v49[2]; // [rsp+30h] [rbp-60h] BYREF
  size_t v50; // [rsp+40h] [rbp-50h] BYREF
  void *v51; // [rsp+48h] [rbp-48h]
  _QWORD v52[8]; // [rsp+50h] [rbp-40h] BYREF

  if ( byte_4F92C88 || !(unsigned int)sub_2207590(&byte_4F92C88) )
    goto LABEL_2;
  srca = v49;
  v50 = 27;
  v9 = (__m128i *)sub_22409D0(&srca, &v50, 0);
  v10 = 1;
  srca = v9;
  v49[0] = v50;
  *v9 = _mm_load_si128((const __m128i *)&xmmword_3F0FCD0);
  qmemcpy(&v9[1], "t_device_sm", 11);
  v48 = v50;
  *((_BYTE *)srca + v50) = 0;
  v11 = &qword_4F92CD0 - 2;
  qword_4F92CA0 = (__int64)&qword_4F92CD0;
  qword_4F92CA8 = 1;
  qword_4F92CB0 = 0;
  qword_4F92CB8 = 0;
  dword_4F92CC0 = 1065353216;
  qword_4F92CC8 = 0;
  qword_4F92CD0 = 0;
  v12 = sub_222D860(&qword_4F92CD0 - 2, 1);
  v14 = v12;
  if ( v12 > qword_4F92CA8 )
  {
    if ( v12 == 1 )
    {
      qword_4F92CD0 = 0;
      v16 = &qword_4F92CD0;
    }
    else
    {
      if ( v12 > 0xFFFFFFFFFFFFFFFLL )
        goto LABEL_60;
      v15 = (void *)sub_22077B0(8 * v12);
      v16 = (__int64 *)memset(v15, 0, 8 * v14);
    }
    qword_4F92CA0 = (__int64)v16;
    qword_4F92CA8 = v14;
  }
  n = sub_22076E0(srca, v48, 3339675911LL);
  v17 = n % qword_4F92CA8;
  v18 = sub_858ED0(&qword_4F92CA0, n % qword_4F92CA8, (__int64)&srca, n);
  if ( !v18 || !*v18 )
  {
    v20 = sub_22077B0(48);
    v21 = (_QWORD *)v20;
    if ( v20 )
      *(_QWORD *)v20 = 0;
    v22 = srca;
    v23 = v48;
    v24 = (_BYTE *)(v20 + 24);
    *(_QWORD *)(v20 + 8) = v20 + 24;
    if ( &v22[v23] && !v22 )
LABEL_3:
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v50 = v23;
    if ( v23 > 0xF )
    {
      v41 = v23;
      v43 = v22;
      v31 = sub_22409D0(v20 + 8, &v50, 0);
      v22 = v43;
      v23 = v41;
      v21[1] = v31;
      v24 = (_BYTE *)v31;
      v21[3] = v50;
    }
    else
    {
      if ( v23 == 1 )
      {
        *(_BYTE *)(v20 + 24) = *v22;
LABEL_33:
        v21[2] = v23;
        v24[v23] = 0;
        v11 = (__int64 *)&dword_4F92CC0;
        v10 = qword_4F92CA8;
        v25 = sub_222DA10(&dword_4F92CC0, qword_4F92CA8, qword_4F92CB8, 1);
        v26 = (__int64 *)qword_4F92CA0;
        v27 = v13;
        if ( !v25 )
        {
LABEL_34:
          v28 = v17;
          v21[5] = n;
          v29 = &v26[v28];
          v30 = (_QWORD *)v26[v28];
          if ( v30 )
          {
            *v21 = *v30;
            *(_QWORD *)*v29 = v21;
          }
          else
          {
            v40 = qword_4F92CB0;
            qword_4F92CB0 = (__int64)v21;
            *v21 = v40;
            if ( v40 )
            {
              v26[*(_QWORD *)(v40 + 40) % (unsigned __int64)qword_4F92CA8] = (__int64)v21;
              v29 = (__int64 *)(v28 * 8 + qword_4F92CA0);
            }
            *v29 = (__int64)&qword_4F92CB0;
          }
          ++qword_4F92CB8;
          goto LABEL_22;
        }
        if ( v13 == 1 )
        {
          qword_4F92CD0 = 0;
          v26 = &qword_4F92CD0;
          goto LABEL_44;
        }
        if ( v13 <= 0xFFFFFFFFFFFFFFFLL )
        {
          v32 = 8 * v13;
          v44 = v13;
          v33 = (void *)sub_22077B0(8 * v13);
          v34 = (__int64 *)memset(v33, 0, v32);
          v27 = v44;
          v26 = v34;
LABEL_44:
          v35 = (_QWORD *)qword_4F92CB0;
          qword_4F92CB0 = 0;
          if ( !v35 )
          {
LABEL_51:
            if ( (__int64 *)qword_4F92CA0 != &qword_4F92CD0 )
            {
              v42 = v27;
              v45 = v26;
              j_j___libc_free_0(qword_4F92CA0, 8 * qword_4F92CA8);
              v27 = v42;
              v26 = v45;
            }
            qword_4F92CA8 = v27;
            qword_4F92CA0 = (__int64)v26;
            v17 = n % v27;
            goto LABEL_34;
          }
          v36 = 0;
          while ( 1 )
          {
            while ( 1 )
            {
              v37 = v35;
              v35 = (_QWORD *)*v35;
              v38 = v37[5] % v27;
              v39 = &v26[v38];
              if ( !*v39 )
                break;
              *v37 = *(_QWORD *)*v39;
              *(_QWORD *)*v39 = v37;
LABEL_47:
              if ( !v35 )
                goto LABEL_51;
            }
            *v37 = qword_4F92CB0;
            qword_4F92CB0 = (__int64)v37;
            *v39 = (__int64)&qword_4F92CB0;
            if ( !*v37 )
            {
              v36 = v38;
              goto LABEL_47;
            }
            v26[v36] = (__int64)v37;
            v36 = v38;
            if ( !v35 )
              goto LABEL_51;
          }
        }
LABEL_60:
        sub_4261EA(v11, v10, v13);
      }
      if ( !v23 )
        goto LABEL_33;
    }
    memcpy(v24, v22, v23);
    v23 = v50;
    v24 = (_BYTE *)v21[1];
    goto LABEL_33;
  }
LABEL_22:
  __cxa_atexit((void (*)(void *))sub_8565C0, &qword_4F92CA0, &qword_4A427C0);
  sub_2207640(&byte_4F92C88);
  if ( srca != v49 )
    j_j___libc_free_0(srca, v49[0] + 1LL);
LABEL_2:
  v50 = (size_t)v52;
  if ( !src )
    goto LABEL_3;
  v2 = (void *)strlen(src);
  srca = v2;
  v3 = (size_t)v2;
  if ( (unsigned __int64)v2 > 0xF )
  {
    v50 = sub_22409D0(&v50, &srca, 0);
    v19 = (_QWORD *)v50;
    v52[0] = srca;
  }
  else
  {
    if ( v2 == (void *)1 )
    {
      LOBYTE(v52[0]) = *src;
      v4 = v52;
      goto LABEL_7;
    }
    if ( !v2 )
    {
      v4 = v52;
      goto LABEL_7;
    }
    v19 = v52;
  }
  memcpy(v19, src, v3);
  v2 = srca;
  v4 = (_QWORD *)v50;
LABEL_7:
  v51 = v2;
  v5 = 0;
  *((_BYTE *)v2 + (_QWORD)v4) = 0;
  v6 = sub_22076E0(v50, v51, 3339675911LL);
  v7 = sub_858ED0(&qword_4F92CA0, v6 % qword_4F92CA8, (__int64)&v50, v6);
  if ( v7 )
    LOBYTE(v5) = *v7 != 0;
  if ( (_QWORD *)v50 != v52 )
    j_j___libc_free_0(v50, v52[0] + 1LL);
  return v5;
}
