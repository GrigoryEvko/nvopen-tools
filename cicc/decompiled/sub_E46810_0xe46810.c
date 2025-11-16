// Function: sub_E46810
// Address: 0xe46810
//
unsigned __int64 *__fastcall sub_E46810(
        unsigned __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        const __m128i *a8,
        unsigned __int64 a9,
        __int64 a10,
        __int64 a11)
{
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  void (__fastcall *v17)(__int64 *, __int64, __int64); // rax
  bool v18; // zf
  unsigned __int64 v19; // rax
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  _QWORD *v22; // r14
  __int64 *v23; // rsi
  __int64 *v24; // rax
  __int64 *v25; // rcx
  __int64 v26; // rax
  __int64 *v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // rax
  void (__fastcall *v32)(__int64 *, __int64, __int64); // rax
  void (__fastcall *v33)(_BYTE *, __int64, __int64); // rax
  void (__fastcall *v34)(_BYTE *, __int64, __int64); // rax
  _QWORD **v35; // r13
  __int64 *v36; // [rsp+8h] [rbp-138h]
  __int64 *v37; // [rsp+20h] [rbp-120h]
  int v38[2]; // [rsp+38h] [rbp-108h] BYREF
  __int64 v39; // [rsp+40h] [rbp-100h] BYREF
  __int64 v40; // [rsp+48h] [rbp-F8h] BYREF
  __int64 v41; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v42; // [rsp+58h] [rbp-E8h] BYREF
  __int64 v43; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v44; // [rsp+68h] [rbp-D8h] BYREF
  __int64 v45; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v46; // [rsp+78h] [rbp-C8h] BYREF
  _QWORD **v47; // [rsp+80h] [rbp-C0h] BYREF
  char v48; // [rsp+88h] [rbp-B8h]
  __int64 v49[2]; // [rsp+90h] [rbp-B0h] BYREF
  __int64 (__fastcall *v50)(__int64 *, __int64, int); // [rsp+A0h] [rbp-A0h]
  __int64 (__fastcall *v51)(__int64); // [rsp+A8h] [rbp-98h]
  char v52; // [rsp+B0h] [rbp-90h]
  _BYTE v53[16]; // [rsp+B8h] [rbp-88h] BYREF
  void (__fastcall *v54)(_BYTE *, _BYTE *, __int64); // [rsp+C8h] [rbp-78h]
  __int64 v55; // [rsp+D0h] [rbp-70h]
  char v56; // [rsp+D8h] [rbp-68h]
  _BYTE v57[16]; // [rsp+E0h] [rbp-60h] BYREF
  void (__fastcall *v58)(_BYTE *, _BYTE *, __int64); // [rsp+F0h] [rbp-50h]
  __int64 v59; // [rsp+F8h] [rbp-48h]
  char v60; // [rsp+100h] [rbp-40h]

  sub_CA08F0((__int64 *)v38, "parse", 5u, (__int64)"Parse IR", 8, byte_4F826E9[0], "irparse", 7u, "LLVM IR Parsing", 15);
  if ( a9 )
  {
    if ( a8->m128i_i8[0] == -34 )
    {
      if ( a8->m128i_i8[1] == -64 && a8->m128i_i8[2] == 23 && a8->m128i_i8[3] == 11 )
      {
LABEL_20:
        v18 = *(_BYTE *)(a4 + 32) == 0;
        v52 = 0;
        if ( v18 )
        {
          v18 = *(_BYTE *)(a4 + 72) == 0;
          v56 = 0;
          if ( v18 )
            goto LABEL_22;
        }
        else
        {
          v50 = 0;
          v32 = *(void (__fastcall **)(__int64 *, __int64, __int64))(a4 + 16);
          if ( v32 )
          {
            v32(v49, a4, 2);
            v51 = *(__int64 (__fastcall **)(__int64))(a4 + 24);
            v50 = *(__int64 (__fastcall **)(__int64 *, __int64, int))(a4 + 16);
          }
          v18 = *(_BYTE *)(a4 + 72) == 0;
          v52 = 1;
          v56 = 0;
          if ( v18 )
          {
LABEL_22:
            v18 = *(_BYTE *)(a4 + 112) == 0;
            v60 = 0;
            if ( v18 )
              goto LABEL_23;
            goto LABEL_58;
          }
        }
        v33 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a4 + 56);
        v54 = 0;
        if ( v33 )
        {
          v33(v53, a4 + 40, 2);
          v55 = *(_QWORD *)(a4 + 64);
          v54 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a4 + 56);
        }
        v18 = *(_BYTE *)(a4 + 112) == 0;
        v56 = 1;
        v60 = 0;
        if ( v18 )
        {
LABEL_23:
          sub_A01950((__int64)&v47, a3, (__int64)v49, v13, v14, v15, a5, a8, a9);
          if ( v60 )
          {
            v60 = 0;
            if ( v58 )
              v58(v57, v57, 3);
          }
          if ( v56 )
          {
            v56 = 0;
            if ( v54 )
              v54(v53, v53, 3);
          }
          if ( v52 )
          {
            v52 = 0;
            if ( v50 )
              v50(v49, (__int64)v49, 3);
          }
          v18 = (v48 & 1) == 0;
          v48 &= ~2u;
          v19 = (unsigned __int64)v47;
          if ( !v18 )
          {
            v47 = 0;
            v20 = v19 | 1;
            v21 = v19 & 0xFFFFFFFFFFFFFFFELL;
            v39 = v20;
            v22 = (_QWORD *)v21;
            if ( v21 )
            {
              v49[0] = a2;
              v23 = (__int64 *)&unk_4F84052;
              v49[1] = (__int64)&a8;
              v39 = 0;
              v40 = 0;
              v41 = 0;
              if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v21 + 48LL))(
                     v21,
                     &unk_4F84052) )
              {
                v24 = (__int64 *)v22[2];
                v25 = (__int64 *)v22[1];
                v42 = 1;
                v36 = v24;
                if ( v25 == v24 )
                {
                  v28 = 1;
                }
                else
                {
                  do
                  {
                    v37 = v25;
                    v44 = *v25;
                    *v25 = 0;
                    sub_E46790(&v45, &v44, v49);
                    v26 = v42;
                    v23 = &v43;
                    v42 = 0;
                    v43 = v26 | 1;
                    sub_9CDB40((unsigned __int64 *)&v46, (unsigned __int64 *)&v43, (unsigned __int64 *)&v45);
                    if ( (v42 & 1) != 0 || (v27 = v37, (v42 & 0xFFFFFFFFFFFFFFFELL) != 0) )
                      sub_C63C30(&v42, (__int64)&v43);
                    v42 |= v46 | 1;
                    if ( (v43 & 1) != 0 || (v43 & 0xFFFFFFFFFFFFFFFELL) != 0 )
                      sub_C63C30(&v43, (__int64)&v43);
                    if ( (v45 & 1) != 0 || (v45 & 0xFFFFFFFFFFFFFFFELL) != 0 )
                      sub_C63C30(&v45, (__int64)&v43);
                    if ( v44 )
                    {
                      (*(void (__fastcall **)(__int64))(*(_QWORD *)v44 + 8LL))(v44);
                      v27 = v37;
                    }
                    v25 = v27 + 1;
                  }
                  while ( v36 != v25 );
                  v28 = v42 | 1;
                }
                v45 = v28;
                (*(void (__fastcall **)(_QWORD *))(*v22 + 8LL))(v22);
              }
              else
              {
                v23 = &v46;
                v46 = (__int64)v22;
                sub_E46790(&v45, &v46, v49);
                if ( v46 )
                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v46 + 8LL))(v46);
              }
              if ( (v45 & 0xFFFFFFFFFFFFFFFELL) != 0 )
                BUG();
              if ( (v41 & 1) != 0 || (v41 & 0xFFFFFFFFFFFFFFFELL) != 0 )
                sub_C63C30(&v41, (__int64)v23);
              if ( (v40 & 1) != 0 || (v40 & 0xFFFFFFFFFFFFFFFELL) != 0 )
                sub_C63C30(&v40, (__int64)v23);
              v31 = v39;
              *a1 = 0;
              if ( (v31 & 1) != 0 || (v31 & 0xFFFFFFFFFFFFFFFELL) != 0 )
                sub_C63C30(&v39, (__int64)v23);
              if ( (v48 & 2) != 0 )
                sub_904700(&v47);
              v35 = v47;
              if ( (v48 & 1) != 0 )
              {
                if ( v47 )
                  ((void (__fastcall *)(_QWORD **))(*v47)[1])(v47);
              }
              else if ( v47 )
              {
                sub_BA9C10(v47, (__int64)v23, v29, v30);
                j_j___libc_free_0(v35, 880);
              }
              goto LABEL_9;
            }
            v19 = 0;
          }
          *a1 = v19;
          goto LABEL_9;
        }
LABEL_58:
        v34 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a4 + 96);
        v58 = 0;
        if ( v34 )
        {
          v34(v57, a4 + 80, 2);
          v59 = *(_QWORD *)(a4 + 104);
          v58 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a4 + 96);
        }
        v60 = 1;
        goto LABEL_23;
      }
    }
    else if ( a8->m128i_i8[0] == 66 && a8->m128i_i8[1] == 67 && a8->m128i_i8[2] == -64 && a8->m128i_i8[3] == -34 )
    {
      goto LABEL_20;
    }
  }
  if ( *(_BYTE *)(a4 + 32) )
  {
    v17 = *(void (__fastcall **)(__int64 *, __int64, __int64))(a4 + 16);
    v50 = 0;
    if ( v17 )
    {
      v17(v49, a4, 2);
      v51 = *(__int64 (__fastcall **)(__int64))(a4 + 24);
      v50 = *(__int64 (__fastcall **)(__int64 *, __int64, int))(a4 + 16);
    }
  }
  else
  {
    v51 = sub_E45A80;
    v50 = sub_E45A90;
  }
  sub_1060120((_DWORD)a1, a2, a3, 0, (unsigned int)sub_E45AA0, (unsigned int)v49, (__int64)a8, a9, a10, a11);
  if ( v50 )
    v50(v49, (__int64)v49, 3);
LABEL_9:
  if ( *(_QWORD *)v38 )
    sub_C9E2A0(*(__int64 *)v38);
  return a1;
}
