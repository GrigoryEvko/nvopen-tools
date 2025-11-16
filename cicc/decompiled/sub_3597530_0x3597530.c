// Function: sub_3597530
// Address: 0x3597530
//
__int64 *__fastcall sub_3597530(__int64 *a1, __int64 a2, __int64 *a3, _QWORD *a4, __int64 a5, __int64 a6)
{
  __int64 *v9; // r13
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // rax
  int v13; // edi
  __int64 v14; // r8
  int v15; // esi
  int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rax
  float v19; // xmm0_4
  __int64 v20; // rax
  void (__fastcall *v21)(__int64 *, const char *); // rbx
  const char *v22; // rax
  const char *v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r10
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdi
  __int64 v33; // [rsp-8h] [rbp-D8h]
  __int64 v34; // [rsp+10h] [rbp-C0h]
  int v35; // [rsp+18h] [rbp-B8h]
  __int64 v36; // [rsp+20h] [rbp-B0h]
  __int64 v37; // [rsp+20h] [rbp-B0h]
  __int64 v38; // [rsp+28h] [rbp-A8h]
  int v39; // [rsp+30h] [rbp-A0h]
  __int64 v40; // [rsp+38h] [rbp-98h]
  _QWORD *v43; // [rsp+50h] [rbp-80h] BYREF
  __int64 v44; // [rsp+58h] [rbp-78h]
  _QWORD v45[2]; // [rsp+60h] [rbp-70h] BYREF
  const char *v46; // [rsp+70h] [rbp-60h] BYREF
  __int64 v47; // [rsp+78h] [rbp-58h]
  _QWORD v48[10]; // [rsp+80h] [rbp-50h] BYREF

  v9 = *(__int64 **)(a2 + 48);
  if ( !v9 )
  {
    if ( qword_503F990 )
    {
      v46 = (const char *)v48;
      sub_3592E00((__int64 *)&v46, (_BYTE *)qword_503F988, qword_503F988 + qword_503F990);
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v47) <= 2
        || (sub_2241490((unsigned __int64 *)&v46, (char *)&off_3F92B2E, 3u),
            v43 = v45,
            sub_3592E00((__int64 *)&v43, (_BYTE *)qword_503F988, qword_503F988 + qword_503F990),
            (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v44) <= 3) )
      {
        sub_4262D8((__int64)"basic_string::append");
      }
      v25 = ".out";
      sub_2241490((unsigned __int64 *)&v43, ".out", 4u);
      v34 = sub_B2BE50(*a3);
      v35 = (int)v43;
      v39 = v44;
      v36 = (__int64)v46;
      v38 = v47;
      v26 = sub_22077B0(0xF8u);
      if ( v26 )
      {
        v28 = v36;
        v37 = v26;
        v25 = (const char *)v34;
        sub_36FEDF0(v26, v34, a2 + 24, (unsigned int)&unk_503F7A0, v35, v39, v28, v38);
        v26 = v37;
        v27 = v33;
      }
      v29 = *(_QWORD *)(a2 + 48);
      *(_QWORD *)(a2 + 48) = v26;
      if ( v29 )
        (*(void (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v29 + 8LL))(v29, v25, v27);
      if ( v43 != v45 )
        j_j___libc_free_0((unsigned __int64)v43);
      if ( v46 != (const char *)v48 )
        j_j___libc_free_0((unsigned __int64)v46);
      v9 = *(__int64 **)(a2 + 48);
    }
    else
    {
      v30 = sub_B2BE50(*a3);
      v47 = 5;
      v40 = v30;
      v46 = "feed_";
      v48[0] = "fetch_";
      v48[1] = 6;
      v48[2] = byte_3F871B3;
      v48[3] = 0;
      v31 = sub_22077B0(0x58u);
      v9 = (__int64 *)v31;
      if ( v31 )
        sub_3597390(v31, v40, (_QWORD *)(a2 + 24), (__int64)"index_to_evict", 14, (__int64)&v46);
      v32 = *(_QWORD *)(a2 + 48);
      *(_QWORD *)(a2 + 48) = 0;
      if ( v32 )
      {
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v32 + 8LL))(v32);
        v9 = *(__int64 **)(a2 + 48);
      }
    }
  }
  v10 = sub_22077B0(0x140u);
  v11 = v10;
  if ( v10 )
  {
    sub_2F40450(v10, (__int64)a3, a4);
    *(_QWORD *)v11 = off_49D8DA8;
    sub_2F40450(v11 + 96, (__int64)a3, a4);
    *(_QWORD *)(v11 + 192) = v9;
    *(_QWORD *)(v11 + 208) = a6;
    *(_QWORD *)(v11 + 216) = 0;
    *(_QWORD *)(v11 + 96) = &unk_4A2AED0;
    *(_QWORD *)(v11 + 200) = a5;
    v12 = a3[4];
    v13 = *(_DWORD *)(v12 + 64);
    if ( v13 )
    {
      v14 = *(_QWORD *)(v12 + 56);
      v15 = 0;
      v16 = 0;
      while ( 1 )
      {
LABEL_7:
        v17 = *(_QWORD *)(v14 + 16LL * (v16 & 0x7FFFFFFF) + 8);
        if ( v17 )
        {
          if ( (*(_BYTE *)(v17 + 4) & 8) != 0 )
          {
            v18 = *(_QWORD *)(v17 + 32);
            if ( !v18 )
              goto LABEL_6;
            while ( (*(_BYTE *)(v18 + 4) & 8) != 0 )
            {
              v18 = *(_QWORD *)(v18 + 32);
              if ( !v18 )
              {
                if ( v13 != ++v16 )
                  goto LABEL_7;
                goto LABEL_13;
              }
            }
          }
          ++v15;
        }
LABEL_6:
        if ( v13 == ++v16 )
        {
LABEL_13:
          v19 = (float)v15;
          goto LABEL_14;
        }
      }
    }
    v19 = 0.0;
LABEL_14:
    *(_QWORD *)(v11 + 232) = 0;
    *(_QWORD *)(v11 + 240) = 0;
    *(_QWORD *)(v11 + 248) = 0;
    *(_DWORD *)(v11 + 256) = 0;
    *(_QWORD *)(v11 + 264) = v11 + 312;
    *(_QWORD *)(v11 + 272) = 1;
    *(_QWORD *)(v11 + 280) = 0;
    *(_QWORD *)(v11 + 288) = 0;
    *(_DWORD *)(v11 + 296) = 1065353216;
    *(_QWORD *)(v11 + 304) = 0;
    *(_QWORD *)(v11 + 312) = 0;
    v20 = *v9;
    *(float *)(v11 + 224) = v19;
    v21 = *(void (__fastcall **)(__int64 *, const char *))(v20 + 16);
    v22 = sub_2E791E0(a3);
    v21(v9, v22);
    *(_QWORD *)(v11 + 216) |= 0x1C0033uLL;
  }
  *a1 = v11;
  return a1;
}
