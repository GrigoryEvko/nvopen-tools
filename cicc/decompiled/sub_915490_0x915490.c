// Function: sub_915490
// Address: 0x915490
//
__int64 __fastcall sub_915490(const char *src)
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
  __int64 v14; // rcx
  __int64 v15; // r14
  void *v16; // rax
  __int64 *v17; // rcx
  unsigned __int64 v18; // r14
  __int64 **v19; // rax
  _QWORD *v20; // rdi
  __int64 v21; // rax
  _QWORD *v22; // r15
  _BYTE *v23; // r9
  size_t v24; // r8
  _BYTE *v25; // rdi
  char v26; // al
  __int64 *v27; // r9
  unsigned __int64 v28; // r8
  __int64 v29; // r14
  __int64 *v30; // rax
  _QWORD *v31; // rdx
  __int64 v32; // rax
  size_t v33; // r14
  void *v34; // rax
  __int64 *v35; // rax
  _QWORD *v36; // rsi
  unsigned __int64 v37; // rdi
  _QWORD *v38; // rcx
  unsigned __int64 v39; // rdx
  __int64 *v40; // rax
  __int64 v41; // rdx
  size_t v42; // [rsp+8h] [rbp-88h]
  unsigned __int64 v43; // [rsp+8h] [rbp-88h]
  _BYTE *v44; // [rsp+10h] [rbp-80h]
  unsigned __int64 v45; // [rsp+10h] [rbp-80h]
  __int64 *v46; // [rsp+10h] [rbp-80h]
  size_t n; // [rsp+18h] [rbp-78h]
  void *srca; // [rsp+20h] [rbp-70h] BYREF
  size_t v49; // [rsp+28h] [rbp-68h]
  _QWORD v50[2]; // [rsp+30h] [rbp-60h] BYREF
  size_t v51; // [rsp+40h] [rbp-50h] BYREF
  void *v52; // [rsp+48h] [rbp-48h]
  _QWORD v53[8]; // [rsp+50h] [rbp-40h] BYREF

  if ( byte_4F6D308 || !(unsigned int)sub_2207590(&byte_4F6D308) )
    goto LABEL_2;
  srca = v50;
  v51 = 27;
  v9 = (__m128i *)sub_22409D0(&srca, &v51, 0);
  v10 = 1;
  srca = v9;
  v50[0] = v51;
  *v9 = _mm_load_si128((const __m128i *)&xmmword_3F0FCD0);
  qmemcpy(&v9[1], "t_device_sm", 11);
  v49 = v51;
  *((_BYTE *)srca + v51) = 0;
  v11 = &qword_4F6D350 - 2;
  qword_4F6D320 = (__int64)&qword_4F6D350;
  qword_4F6D328 = 1;
  qword_4F6D330 = 0;
  qword_4F6D338 = 0;
  dword_4F6D340 = 1065353216;
  qword_4F6D348 = 0;
  qword_4F6D350 = 0;
  v12 = sub_222D860(&qword_4F6D350 - 2, 1);
  v15 = v12;
  if ( v12 > qword_4F6D328 )
  {
    if ( v12 == 1 )
    {
      qword_4F6D350 = 0;
      v17 = &qword_4F6D350;
    }
    else
    {
      if ( v12 > 0xFFFFFFFFFFFFFFFLL )
        goto LABEL_60;
      v16 = (void *)sub_22077B0(8 * v12);
      v17 = (__int64 *)memset(v16, 0, 8 * v15);
    }
    qword_4F6D320 = (__int64)v17;
    qword_4F6D328 = v15;
  }
  n = sub_22076E0(srca, v49, 3339675911LL);
  v18 = n % qword_4F6D328;
  v19 = sub_858ED0(&qword_4F6D320, n % qword_4F6D328, (__int64)&srca, n);
  if ( !v19 || !*v19 )
  {
    v21 = sub_22077B0(48);
    v22 = (_QWORD *)v21;
    if ( v21 )
      *(_QWORD *)v21 = 0;
    v23 = srca;
    v24 = v49;
    v25 = (_BYTE *)(v21 + 24);
    *(_QWORD *)(v21 + 8) = v21 + 24;
    if ( &v23[v24] && !v23 )
LABEL_3:
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v51 = v24;
    if ( v24 > 0xF )
    {
      v42 = v24;
      v44 = v23;
      v32 = sub_22409D0(v21 + 8, &v51, 0);
      v23 = v44;
      v24 = v42;
      v22[1] = v32;
      v25 = (_BYTE *)v32;
      v22[3] = v51;
    }
    else
    {
      if ( v24 == 1 )
      {
        *(_BYTE *)(v21 + 24) = *v23;
LABEL_33:
        v22[2] = v24;
        v25[v24] = 0;
        v11 = (__int64 *)&dword_4F6D340;
        v10 = qword_4F6D328;
        v26 = sub_222DA10(&dword_4F6D340, qword_4F6D328, qword_4F6D338, 1);
        v27 = (__int64 *)qword_4F6D320;
        v28 = v13;
        if ( !v26 )
        {
LABEL_34:
          v29 = v18;
          v22[5] = n;
          v30 = &v27[v29];
          v31 = (_QWORD *)v27[v29];
          if ( v31 )
          {
            *v22 = *v31;
            *(_QWORD *)*v30 = v22;
          }
          else
          {
            v41 = qword_4F6D330;
            qword_4F6D330 = (__int64)v22;
            *v22 = v41;
            if ( v41 )
            {
              v27[*(_QWORD *)(v41 + 40) % (unsigned __int64)qword_4F6D328] = (__int64)v22;
              v30 = (__int64 *)(v29 * 8 + qword_4F6D320);
            }
            *v30 = (__int64)&qword_4F6D330;
          }
          ++qword_4F6D338;
          goto LABEL_22;
        }
        if ( v13 == 1 )
        {
          qword_4F6D350 = 0;
          v27 = &qword_4F6D350;
          goto LABEL_44;
        }
        if ( v13 <= 0xFFFFFFFFFFFFFFFLL )
        {
          v33 = 8 * v13;
          v45 = v13;
          v34 = (void *)sub_22077B0(8 * v13);
          v35 = (__int64 *)memset(v34, 0, v33);
          v28 = v45;
          v27 = v35;
LABEL_44:
          v36 = (_QWORD *)qword_4F6D330;
          qword_4F6D330 = 0;
          if ( !v36 )
          {
LABEL_51:
            if ( (__int64 *)qword_4F6D320 != &qword_4F6D350 )
            {
              v43 = v28;
              v46 = v27;
              j_j___libc_free_0(qword_4F6D320, 8 * qword_4F6D328);
              v28 = v43;
              v27 = v46;
            }
            qword_4F6D328 = v28;
            qword_4F6D320 = (__int64)v27;
            v18 = n % v28;
            goto LABEL_34;
          }
          v37 = 0;
          while ( 1 )
          {
            while ( 1 )
            {
              v38 = v36;
              v36 = (_QWORD *)*v36;
              v39 = v38[5] % v28;
              v40 = &v27[v39];
              if ( !*v40 )
                break;
              *v38 = *(_QWORD *)*v40;
              *(_QWORD *)*v40 = v38;
LABEL_47:
              if ( !v36 )
                goto LABEL_51;
            }
            *v38 = qword_4F6D330;
            qword_4F6D330 = (__int64)v38;
            *v40 = (__int64)&qword_4F6D330;
            if ( !*v38 )
            {
              v37 = v39;
              goto LABEL_47;
            }
            v27[v37] = (__int64)v38;
            v37 = v39;
            if ( !v36 )
              goto LABEL_51;
          }
        }
LABEL_60:
        sub_4261EA(v11, v10, v13, v14);
      }
      if ( !v24 )
        goto LABEL_33;
    }
    memcpy(v25, v23, v24);
    v24 = v51;
    v25 = (_BYTE *)v22[1];
    goto LABEL_33;
  }
LABEL_22:
  __cxa_atexit((void (*)(void *))sub_8565C0, &qword_4F6D320, &qword_4A427C0);
  sub_2207640(&byte_4F6D308);
  if ( srca != v50 )
    j_j___libc_free_0(srca, v50[0] + 1LL);
LABEL_2:
  v51 = (size_t)v53;
  if ( !src )
    goto LABEL_3;
  v2 = (void *)strlen(src);
  srca = v2;
  v3 = (size_t)v2;
  if ( (unsigned __int64)v2 > 0xF )
  {
    v51 = sub_22409D0(&v51, &srca, 0);
    v20 = (_QWORD *)v51;
    v53[0] = srca;
  }
  else
  {
    if ( v2 == (void *)1 )
    {
      LOBYTE(v53[0]) = *src;
      v4 = v53;
      goto LABEL_7;
    }
    if ( !v2 )
    {
      v4 = v53;
      goto LABEL_7;
    }
    v20 = v53;
  }
  memcpy(v20, src, v3);
  v2 = srca;
  v4 = (_QWORD *)v51;
LABEL_7:
  v52 = v2;
  v5 = 0;
  *((_BYTE *)v2 + (_QWORD)v4) = 0;
  v6 = sub_22076E0(v51, v52, 3339675911LL);
  v7 = sub_858ED0(&qword_4F6D320, v6 % qword_4F6D328, (__int64)&v51, v6);
  if ( v7 )
    LOBYTE(v5) = *v7 != 0;
  if ( (_QWORD *)v51 != v53 )
    j_j___libc_free_0(v51, v53[0] + 1LL);
  return v5;
}
