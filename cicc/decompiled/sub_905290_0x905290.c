// Function: sub_905290
// Address: 0x905290
//
__int64 __fastcall sub_905290(const char *src, __int64 a2, char a3)
{
  size_t v5; // r15
  __int64 v6; // rax
  __int64 v7; // r14
  unsigned int v8; // r15d
  __int64 v9; // rax
  __int64 v11; // rsi
  int v12; // ebx
  __int64 v13; // rdx
  __int64 v14; // r13
  void *v15; // rdx
  __int64 v16; // r13
  void *v17; // rdx
  size_t v18; // rax
  _WORD *v19; // rdi
  size_t v20; // r15
  unsigned __int64 v21; // rax
  __int64 v22; // rax
  size_t v23; // rax
  __m128i *v24; // rdi
  size_t v25; // r14
  unsigned __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  __m128i si128; // xmm0
  char *v30; // rdi
  __int64 v31; // rax
  unsigned int v33; // [rsp+10h] [rbp-250h] BYREF
  __int64 v34; // [rsp+18h] [rbp-248h]
  __int64 (__fastcall **v35)(); // [rsp+20h] [rbp-240h] BYREF
  __int64 v36; // [rsp+28h] [rbp-238h]
  _QWORD v37[7]; // [rsp+30h] [rbp-230h] BYREF
  _BYTE v38[48]; // [rsp+68h] [rbp-1F8h] BYREF
  _BYTE v39[136]; // [rsp+98h] [rbp-1C8h] BYREF
  _QWORD v40[4]; // [rsp+120h] [rbp-140h] BYREF
  int v41; // [rsp+140h] [rbp-120h]
  __int64 v42; // [rsp+1F8h] [rbp-68h]
  __int16 v43; // [rsp+200h] [rbp-60h]
  __int64 v44; // [rsp+208h] [rbp-58h]
  __int64 v45; // [rsp+210h] [rbp-50h]
  __int64 v46; // [rsp+218h] [rbp-48h]
  __int64 v47; // [rsp+220h] [rbp-40h]

  if ( byte_4F6D2E0 || !(unsigned int)sub_2207590(&byte_4F6D2E0) )
  {
    if ( a3 )
      goto LABEL_3;
  }
  else
  {
    qword_4F6D2E8 = sub_CB72A0();
    sub_2207640(&byte_4F6D2E0);
    if ( a3 )
      goto LABEL_3;
  }
  sub_222DF20(v40);
  v42 = 0;
  v43 = 0;
  v40[0] = off_4A06798;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v35 = (__int64 (__fastcall **)())qword_4A064D0;
  v47 = 0;
  *(__int64 (__fastcall ***)())((char *)&v35 + qword_4A064D0[-3]) = (__int64 (__fastcall **)())&unk_4A064F8;
  v36 = 0;
  sub_222DD70((char *)&v35 + (_QWORD)*(v35 - 3), 0);
  v35 = off_4A06540;
  v40[0] = off_4A06568;
  sub_222BA80(v37);
  sub_222DD70(v40, v37);
  if ( sub_222C940(v37, src, 8) )
  {
    v11 = 0;
    sub_222DC80((char *)&v35 + (_QWORD)*(v35 - 3), 0);
  }
  else
  {
    v30 = (char *)&v35 + (_QWORD)*(v35 - 3);
    v11 = *((_DWORD *)v30 + 8) | 4u;
    sub_222DC80(v30, v11);
  }
  v35 = off_4A06540;
  v40[0] = off_4A06568;
  v12 = v41 & 5;
  v37[0] = off_4A06448;
  sub_222C7F0(v37);
  sub_2207D90(v39);
  v37[0] = off_4A07480;
  sub_2209150(v38, v11, v13);
  v35 = (__int64 (__fastcall **)())qword_4A064D0;
  *(__int64 (__fastcall ***)())((char *)&v35 + qword_4A064D0[-3]) = (__int64 (__fastcall **)())&unk_4A064F8;
  v36 = 0;
  v40[0] = off_4A06798;
  sub_222E050(v40);
  if ( !v12 )
  {
    v14 = qword_4F6D2E8;
    v15 = *(void **)(qword_4F6D2E8 + 32);
    if ( *(_QWORD *)(qword_4F6D2E8 + 24) - (_QWORD)v15 <= 0xEu )
    {
      v14 = sub_CB6200(qword_4F6D2E8, "Error opening '", 15);
    }
    else
    {
      qmemcpy(v15, "Error opening '", 15);
      *(_QWORD *)(v14 + 32) += 15LL;
    }
    if ( src )
    {
      v23 = strlen(src);
      v24 = *(__m128i **)(v14 + 32);
      v25 = v23;
      v26 = *(_QWORD *)(v14 + 24) - (_QWORD)v24;
      if ( v25 <= v26 )
      {
        if ( v25 )
        {
          memcpy(v24, src, v25);
          v27 = *(_QWORD *)(v14 + 24);
          v24 = (__m128i *)(v25 + *(_QWORD *)(v14 + 32));
          *(_QWORD *)(v14 + 32) = v24;
          v26 = v27 - (_QWORD)v24;
        }
        goto LABEL_34;
      }
      v14 = sub_CB6200(v14, src, v25);
    }
    v24 = *(__m128i **)(v14 + 32);
    v26 = *(_QWORD *)(v14 + 24) - (_QWORD)v24;
LABEL_34:
    if ( v26 <= 0xF )
    {
      v31 = sub_CB6200(v14, "': file exists!\n", 16);
      v28 = *(_QWORD *)(v31 + 32);
      v14 = v31;
    }
    else
    {
      *v24 = _mm_load_si128((const __m128i *)&xmmword_3F0F580);
      v28 = *(_QWORD *)(v14 + 32) + 16LL;
      *(_QWORD *)(v14 + 32) = v28;
    }
    if ( (unsigned __int64)(*(_QWORD *)(v14 + 24) - v28) <= 0x2C )
    {
      sub_CB6200(v14, "Use -f command line argument to force output\n", 45);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F0F590);
      qmemcpy((void *)(v28 + 32), "force output\n", 13);
      *(__m128i *)v28 = si128;
      *(__m128i *)(v28 + 16) = _mm_load_si128((const __m128i *)&xmmword_3F0F5A0);
      *(_QWORD *)(v14 + 32) += 45LL;
    }
    return 1;
  }
LABEL_3:
  v33 = 0;
  v5 = 0;
  v34 = sub_2241E40();
  if ( src )
    v5 = strlen(src);
  v6 = sub_22077B0(96);
  v7 = v6;
  if ( v6 )
    sub_CB7060(v6, src, v5, &v33, 0);
  v8 = v33;
  if ( !v33 )
  {
    sub_B848C0(&v35);
    v9 = sub_A3CDF0(v7, 1);
    sub_B8B500(&v35, v9, 0);
    sub_B89FE0(&v35, a2);
    sub_B82680(&v35);
    if ( v7 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
    return v8;
  }
  v16 = qword_4F6D2E8;
  v17 = *(void **)(qword_4F6D2E8 + 32);
  if ( *(_QWORD *)(qword_4F6D2E8 + 24) - (_QWORD)v17 <= 0xDu )
  {
    v16 = sub_CB6200(qword_4F6D2E8, "Error opening ", 14);
  }
  else
  {
    qmemcpy(v17, "Error opening ", 14);
    *(_QWORD *)(v16 + 32) += 14LL;
  }
  if ( src )
  {
    v18 = strlen(src);
    v19 = *(_WORD **)(v16 + 32);
    v20 = v18;
    v21 = *(_QWORD *)(v16 + 24) - (_QWORD)v19;
    if ( v20 <= v21 )
    {
      if ( v20 )
      {
        memcpy(v19, src, v20);
        v22 = *(_QWORD *)(v16 + 24);
        v19 = (_WORD *)(v20 + *(_QWORD *)(v16 + 32));
        *(_QWORD *)(v16 + 32) = v19;
        v21 = v22 - (_QWORD)v19;
      }
      goto LABEL_24;
    }
    v16 = sub_CB6200(v16, src, v20);
  }
  v19 = *(_WORD **)(v16 + 32);
  v21 = *(_QWORD *)(v16 + 24) - (_QWORD)v19;
LABEL_24:
  if ( v21 <= 1 )
  {
    sub_CB6200(v16, "!\n", 2);
  }
  else
  {
    *v19 = 2593;
    *(_QWORD *)(v16 + 32) += 2LL;
  }
  if ( !v7 )
    return 1;
  v8 = 1;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
  return v8;
}
