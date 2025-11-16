// Function: sub_31DBC20
// Address: 0x31dbc20
//
void __fastcall sub_31DBC20(__int64 a1, __int64 *a2)
{
  __int64 v2; // r13
  __int64 v3; // r14
  unsigned __int64 v6; // r15
  _BYTE *v7; // rax
  _BYTE *v8; // r13
  __int64 v9; // r8
  unsigned __int8 v10; // al
  _BYTE **v11; // rdx
  _BYTE *v12; // rdi
  __int64 v13; // r8
  __int64 v14; // rax
  _BYTE *v15; // rdi
  __int64 v16; // r13
  unsigned __int8 *v17; // rsi
  size_t v18; // rdx
  unsigned __int64 v19; // rax
  const char *v20; // rax
  size_t v21; // rdx
  _BYTE *v22; // rdi
  unsigned __int8 *v23; // rsi
  unsigned __int64 v24; // rax
  size_t v25; // r12
  __int64 v26; // rdi
  _BYTE *v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rdx
  unsigned __int64 v30; // rax
  unsigned __int8 v31; // dl
  __int64 *v32; // rax
  void *v33; // rax
  size_t v34; // rdx
  __int64 v35; // rax
  __int64 (__fastcall **v36)(); // rax
  _BYTE *v37; // rsi
  __int64 v38; // rax
  __int64 v39; // r13
  __int64 v40; // rdi
  _QWORD *v41; // rax
  __m128i *v42; // rdx
  __int64 v43; // r12
  __m128i si128; // xmm0
  __int64 v45; // rax
  size_t v46; // rdx
  char *v47; // rsi
  _BYTE *v48; // rdx
  __int64 v49; // [rsp+10h] [rbp-70h]
  __int64 v50; // [rsp+10h] [rbp-70h]
  __int64 v51; // [rsp+10h] [rbp-70h]
  size_t v52; // [rsp+18h] [rbp-68h]
  size_t v53; // [rsp+18h] [rbp-68h]
  unsigned int v54; // [rsp+20h] [rbp-60h] BYREF
  __int64 (__fastcall **v55)(); // [rsp+28h] [rbp-58h]
  unsigned __int8 *v56[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v57; // [rsp+40h] [rbp-40h] BYREF

  v2 = a2[1];
  if ( !*(_QWORD *)(v2 + 920) )
    return;
  v3 = a2[6];
  v6 = *(_QWORD *)(v3 + 48) + *(_QWORD *)(v3 + 688);
  if ( *(_QWORD *)(a1 + 496) )
    goto LABEL_3;
  v50 = *(_QWORD *)(v2 + 920);
  v54 = 0;
  v36 = sub_2241E40();
  v37 = *(_BYTE **)(v2 + 912);
  v55 = v36;
  v38 = sub_22077B0(0x60u);
  v39 = v38;
  if ( v38 )
    sub_CB7060(v38, v37, v50, (__int64)&v54, 1u);
  v40 = *(_QWORD *)(a1 + 496);
  *(_QWORD *)(a1 + 496) = v39;
  if ( v40 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v40 + 8LL))(v40);
  if ( !v54 )
  {
LABEL_3:
    v7 = (_BYTE *)sub_B92180(*a2);
    v8 = v7;
    if ( v7 )
    {
      v9 = *(_QWORD *)(a1 + 496);
      if ( (*v7 == 16
         || ((v10 = *(v7 - 16), (v10 & 2) != 0)
           ? (v11 = (_BYTE **)*((_QWORD *)v8 - 4))
           : (v11 = (_BYTE **)&v8[-8 * ((v10 >> 2) & 0xF) - 16]),
             (v7 = *v11) != 0))
        && ((v31 = *(v7 - 16), (v31 & 2) == 0)
          ? (v32 = (__int64 *)&v7[-8 * ((v31 >> 2) & 0xF) - 16])
          : (v32 = (__int64 *)*((_QWORD *)v7 - 4)),
            *v32) )
      {
        v49 = *(_QWORD *)(a1 + 496);
        v33 = (void *)sub_B91420(*v32);
        v9 = v49;
        v12 = *(_BYTE **)(v49 + 32);
        if ( *(_QWORD *)(v49 + 24) - (_QWORD)v12 >= v34 )
        {
          if ( v34 )
          {
            v52 = v34;
            memcpy(v12, v33, v34);
            v9 = v49;
            v48 = (_BYTE *)(*(_QWORD *)(v49 + 32) + v52);
            *(_QWORD *)(v49 + 32) = v48;
            v12 = v48;
          }
        }
        else
        {
          v35 = sub_CB6200(v49, (unsigned __int8 *)v33, v34);
          v12 = *(_BYTE **)(v35 + 32);
          v9 = v35;
        }
      }
      else
      {
        v12 = *(_BYTE **)(*(_QWORD *)(a1 + 496) + 32LL);
      }
      if ( *(_QWORD *)(v9 + 24) <= (unsigned __int64)v12 )
      {
        v9 = sub_CB5D20(v9, 58);
      }
      else
      {
        *(_QWORD *)(v9 + 32) = v12 + 1;
        *v12 = 58;
      }
      sub_CB59D0(v9, *((unsigned int *)v8 + 4));
      v13 = *(_QWORD *)(a1 + 496);
      v15 = *(_BYTE **)(v13 + 32);
      v16 = v13;
      if ( (unsigned __int64)v15 >= *(_QWORD *)(v13 + 24) )
      {
LABEL_33:
        v16 = sub_CB5D20(v13, 58);
LABEL_14:
        v20 = sub_2E791E0(a2);
        v22 = *(_BYTE **)(v16 + 32);
        v23 = (unsigned __int8 *)v20;
        v24 = *(_QWORD *)(v16 + 24);
        v25 = v21;
        if ( v21 > v24 - (unsigned __int64)v22 )
        {
          v45 = sub_CB6200(v16, v23, v21);
          v22 = *(_BYTE **)(v45 + 32);
          v16 = v45;
          if ( (unsigned __int64)v22 < *(_QWORD *)(v45 + 24) )
          {
LABEL_18:
            *(_QWORD *)(v16 + 32) = v22 + 1;
            *v22 = 9;
            goto LABEL_19;
          }
        }
        else
        {
          if ( v21 )
          {
            memcpy(v22, v23, v21);
            v24 = *(_QWORD *)(v16 + 24);
            v22 = (_BYTE *)(v25 + *(_QWORD *)(v16 + 32));
            *(_QWORD *)(v16 + 32) = v22;
          }
          if ( (unsigned __int64)v22 < v24 )
            goto LABEL_18;
        }
        v16 = sub_CB5D20(v16, 9);
LABEL_19:
        v26 = sub_CB59D0(v16, v6);
        v27 = *(_BYTE **)(v26 + 32);
        if ( (unsigned __int64)v27 >= *(_QWORD *)(v26 + 24) )
        {
          sub_CB5D20(v26, 9);
        }
        else
        {
          *(_QWORD *)(v26 + 32) = v27 + 1;
          *v27 = 9;
        }
        v28 = *(_QWORD *)(a1 + 496);
        v29 = *(_QWORD *)(v28 + 32);
        v30 = *(_QWORD *)(v28 + 24) - v29;
        if ( *(_BYTE *)(v3 + 36) )
        {
          if ( v30 > 7 )
          {
            *(_QWORD *)v29 = 0xA63696D616E7964LL;
            *(_QWORD *)(v28 + 32) += 8LL;
            return;
          }
          v46 = 8;
          v47 = "dynamic\n";
        }
        else
        {
          if ( v30 > 6 )
          {
            *(_DWORD *)v29 = 1952543859;
            *(_WORD *)(v29 + 4) = 25449;
            *(_BYTE *)(v29 + 6) = 10;
            *(_QWORD *)(v28 + 32) += 7LL;
            return;
          }
          v46 = 7;
          v47 = "static\n";
        }
        sub_CB6200(v28, (unsigned __int8 *)v47, v46);
        return;
      }
    }
    else
    {
      v13 = *(_QWORD *)(a1 + 496);
      v14 = *(_QWORD *)(*a2 + 40);
      v15 = *(_BYTE **)(v13 + 32);
      v16 = v13;
      v17 = *(unsigned __int8 **)(v14 + 168);
      v18 = *(_QWORD *)(v14 + 176);
      v19 = *(_QWORD *)(v13 + 24);
      if ( v18 > v19 - (unsigned __int64)v15 )
      {
        sub_CB6200(*(_QWORD *)(a1 + 496), v17, v18);
        v13 = *(_QWORD *)(a1 + 496);
        v15 = *(_BYTE **)(v13 + 32);
        v19 = *(_QWORD *)(v13 + 24);
        v16 = v13;
      }
      else if ( v18 )
      {
        v51 = *(_QWORD *)(a1 + 496);
        v53 = v18;
        memcpy(v15, v17, v18);
        *(_QWORD *)(v51 + 32) += v53;
        v13 = *(_QWORD *)(a1 + 496);
        v15 = *(_BYTE **)(v13 + 32);
        v19 = *(_QWORD *)(v13 + 24);
        v16 = v13;
      }
      if ( (unsigned __int64)v15 >= v19 )
        goto LABEL_33;
    }
    *(_QWORD *)(v13 + 32) = v15 + 1;
    *v15 = 58;
    goto LABEL_14;
  }
  v41 = sub_CB72A0();
  v42 = (__m128i *)v41[4];
  v43 = (__int64)v41;
  if ( v41[3] - (_QWORD)v42 <= 0x14u )
  {
    v43 = sub_CB6200((__int64)v41, "Could not open file: ", 0x15u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_439AB70);
    v42[1].m128i_i32[0] = 979725417;
    v42[1].m128i_i8[4] = 32;
    *v42 = si128;
    v41[4] += 21LL;
  }
  (*((void (__fastcall **)(unsigned __int8 **, __int64 (__fastcall **)(), _QWORD))*v55 + 4))(v56, v55, v54);
  sub_CB6200(v43, v56[0], (size_t)v56[1]);
  if ( (__int64 *)v56[0] != &v57 )
    j_j___libc_free_0((unsigned __int64)v56[0]);
}
