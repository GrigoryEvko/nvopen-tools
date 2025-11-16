// Function: sub_1264D50
// Address: 0x1264d50
//
__int64 __fastcall sub_1264D50(char *src, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  const char *v5; // r12
  char v6; // bl
  size_t v7; // r15
  __int64 v8; // rax
  __int64 v9; // r14
  unsigned int v10; // r15d
  __int64 v11; // rax
  int v13; // ebx
  __int64 v14; // rdx
  __int64 v15; // r13
  void *v16; // rdx
  __int64 v17; // r13
  void *v18; // rdx
  size_t v19; // rax
  _WORD *v20; // rdi
  size_t v21; // r15
  unsigned __int64 v22; // rax
  __int64 v23; // rax
  size_t v24; // rax
  __m128i *v25; // rdi
  size_t v26; // r14
  unsigned __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  __m128i si128; // xmm0
  char *v31; // rdi
  __int64 v32; // rax
  __int64 v33; // [rsp+8h] [rbp-258h]
  unsigned int v34; // [rsp+10h] [rbp-250h] BYREF
  __int64 v35; // [rsp+18h] [rbp-248h]
  __int64 (__fastcall **v36)(); // [rsp+20h] [rbp-240h] BYREF
  __int64 v37; // [rsp+28h] [rbp-238h]
  _QWORD v38[7]; // [rsp+30h] [rbp-230h] BYREF
  _BYTE v39[48]; // [rsp+68h] [rbp-1F8h] BYREF
  _BYTE v40[136]; // [rsp+98h] [rbp-1C8h] BYREF
  _QWORD v41[4]; // [rsp+120h] [rbp-140h] BYREF
  int v42; // [rsp+140h] [rbp-120h]
  __int64 v43; // [rsp+1F8h] [rbp-68h]
  __int16 v44; // [rsp+200h] [rbp-60h]
  __int64 v45; // [rsp+208h] [rbp-58h]
  __int64 v46; // [rsp+210h] [rbp-50h]
  __int64 v47; // [rsp+218h] [rbp-48h]
  __int64 v48; // [rsp+220h] [rbp-40h]

  v5 = src;
  v6 = a3;
  v33 = a2;
  if ( byte_4F92C60 || (src = &byte_4F92C60, !(unsigned int)sub_2207590(&byte_4F92C60)) )
  {
    if ( v6 )
      goto LABEL_3;
  }
  else
  {
    src = &byte_4F92C60;
    qword_4F92C68 = sub_16E8CB0(&byte_4F92C60, a2, a3);
    sub_2207640(&byte_4F92C60);
    if ( v6 )
      goto LABEL_3;
  }
  sub_222DF20(v41);
  v43 = 0;
  v44 = 0;
  v41[0] = off_4A06798;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v36 = (__int64 (__fastcall **)())qword_4A064D0;
  v48 = 0;
  *(__int64 (__fastcall ***)())((char *)&v36 + qword_4A064D0[-3]) = (__int64 (__fastcall **)())&unk_4A064F8;
  v37 = 0;
  sub_222DD70((char *)&v36 + (_QWORD)*(v36 - 3), 0);
  v36 = off_4A06540;
  v41[0] = off_4A06568;
  sub_222BA80(v38);
  sub_222DD70(v41, v38);
  if ( sub_222C940(v38, v5, 8) )
  {
    a2 = 0;
    sub_222DC80((char *)&v36 + (_QWORD)*(v36 - 3), 0);
  }
  else
  {
    v31 = (char *)&v36 + (_QWORD)*(v36 - 3);
    a2 = *((_DWORD *)v31 + 8) | 4u;
    sub_222DC80(v31, a2);
  }
  v36 = off_4A06540;
  v41[0] = off_4A06568;
  v13 = v42 & 5;
  v38[0] = off_4A06448;
  sub_222C7F0(v38);
  sub_2207D90(v40);
  v38[0] = off_4A07480;
  sub_2209150(v39, a2, v14);
  v36 = (__int64 (__fastcall **)())qword_4A064D0;
  src = (char *)v41;
  *(__int64 (__fastcall ***)())((char *)&v36 + qword_4A064D0[-3]) = (__int64 (__fastcall **)())&unk_4A064F8;
  v37 = 0;
  v41[0] = off_4A06798;
  sub_222E050(v41);
  if ( !v13 )
  {
    v15 = qword_4F92C68;
    v16 = *(void **)(qword_4F92C68 + 24);
    if ( *(_QWORD *)(qword_4F92C68 + 16) - (_QWORD)v16 <= 0xEu )
    {
      v15 = sub_16E7EE0(qword_4F92C68, "Error opening '", 15);
    }
    else
    {
      qmemcpy(v16, "Error opening '", 15);
      *(_QWORD *)(v15 + 24) += 15LL;
    }
    if ( v5 )
    {
      v24 = strlen(v5);
      v25 = *(__m128i **)(v15 + 24);
      v26 = v24;
      v27 = *(_QWORD *)(v15 + 16) - (_QWORD)v25;
      if ( v26 <= v27 )
      {
        if ( v26 )
        {
          memcpy(v25, v5, v26);
          v28 = *(_QWORD *)(v15 + 16);
          v25 = (__m128i *)(v26 + *(_QWORD *)(v15 + 24));
          *(_QWORD *)(v15 + 24) = v25;
          v27 = v28 - (_QWORD)v25;
        }
        goto LABEL_34;
      }
      v15 = sub_16E7EE0(v15, v5, v26);
    }
    v25 = *(__m128i **)(v15 + 24);
    v27 = *(_QWORD *)(v15 + 16) - (_QWORD)v25;
LABEL_34:
    if ( v27 <= 0xF )
    {
      v32 = sub_16E7EE0(v15, "': file exists!\n", 16);
      v29 = *(_QWORD *)(v32 + 24);
      v15 = v32;
    }
    else
    {
      *v25 = _mm_load_si128((const __m128i *)&xmmword_3F0F580);
      v29 = *(_QWORD *)(v15 + 24) + 16LL;
      *(_QWORD *)(v15 + 24) = v29;
    }
    if ( (unsigned __int64)(*(_QWORD *)(v15 + 16) - v29) <= 0x2C )
    {
      sub_16E7EE0(v15, "Use -f command line argument to force output\n", 45);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F0F590);
      qmemcpy((void *)(v29 + 32), "force output\n", 13);
      *(__m128i *)v29 = si128;
      *(__m128i *)(v29 + 16) = _mm_load_si128((const __m128i *)&xmmword_3F0F5A0);
      *(_QWORD *)(v15 + 24) += 45LL;
    }
    return 1;
  }
LABEL_3:
  v34 = 0;
  v7 = 0;
  v35 = sub_2241E40(src, a2, a3, a4, a5);
  if ( v5 )
    v7 = strlen(v5);
  v8 = sub_22077B0(80);
  v9 = v8;
  if ( v8 )
    sub_16E8AF0(v8, v5, v7, &v34, 0);
  v10 = v34;
  if ( !v34 )
  {
    sub_1611EE0(&v36);
    v11 = sub_153C7C0(v9, 1, 0, 0);
    sub_1619140(&v36, v11, 0);
    sub_1619BD0(&v36, v33);
    sub_160FE50(&v36);
    if ( v9 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v9 + 8LL))(v9);
    return v10;
  }
  v17 = qword_4F92C68;
  v18 = *(void **)(qword_4F92C68 + 24);
  if ( *(_QWORD *)(qword_4F92C68 + 16) - (_QWORD)v18 <= 0xDu )
  {
    v17 = sub_16E7EE0(qword_4F92C68, "Error opening ", 14);
  }
  else
  {
    qmemcpy(v18, "Error opening ", 14);
    *(_QWORD *)(v17 + 24) += 14LL;
  }
  if ( v5 )
  {
    v19 = strlen(v5);
    v20 = *(_WORD **)(v17 + 24);
    v21 = v19;
    v22 = *(_QWORD *)(v17 + 16) - (_QWORD)v20;
    if ( v21 <= v22 )
    {
      if ( v21 )
      {
        memcpy(v20, v5, v21);
        v23 = *(_QWORD *)(v17 + 16);
        v20 = (_WORD *)(v21 + *(_QWORD *)(v17 + 24));
        *(_QWORD *)(v17 + 24) = v20;
        v22 = v23 - (_QWORD)v20;
      }
      goto LABEL_24;
    }
    v17 = sub_16E7EE0(v17, v5, v21);
  }
  v20 = *(_WORD **)(v17 + 24);
  v22 = *(_QWORD *)(v17 + 16) - (_QWORD)v20;
LABEL_24:
  if ( v22 <= 1 )
  {
    sub_16E7EE0(v17, "!\n", 2);
  }
  else
  {
    *v20 = 2593;
    *(_QWORD *)(v17 + 24) += 2LL;
  }
  if ( !v9 )
    return 1;
  v10 = 1;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v9 + 8LL))(v9);
  return v10;
}
