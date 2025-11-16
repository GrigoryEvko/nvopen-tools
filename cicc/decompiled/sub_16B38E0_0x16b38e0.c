// Function: sub_16B38E0
// Address: 0x16b38e0
//
__int64 __fastcall sub_16B38E0(
        unsigned __int8 (__fastcall ***a1)(_QWORD, _QWORD),
        __int64 a2,
        unsigned __int8 (__fastcall ***a3)(_QWORD, __int64),
        unsigned __int8 (__fastcall ***a4)(_QWORD, const char *),
        int a5)
{
  __int64 v10; // rax
  __int64 v11; // rax
  size_t v12; // rdx
  const char *v13; // rsi
  void *v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rsi
  unsigned __int8 (__fastcall ***v17)(_QWORD, _QWORD); // rdi
  __int64 v18; // rdx
  unsigned int v19; // r13d
  unsigned __int8 (__fastcall *v20)(_QWORD, __int64); // r15
  __int64 v21; // rax
  __int64 v23; // rax
  __int64 v24; // rbx
  const char *v25; // rax
  size_t v26; // rdx
  __int64 v27; // rsi
  unsigned int v28; // r13d
  unsigned __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rsi
  unsigned int v32; // r13d
  __int64 v33; // rax
  unsigned __int8 (__fastcall *v34)(_QWORD, const char *); // rbx
  __int64 v35; // rdi
  const char *v36; // rsi
  __int64 v37; // rdx
  __int64 v38; // rax
  size_t v39; // rdx
  __int64 v40; // [rsp+0h] [rbp-40h]
  int v41; // [rsp+8h] [rbp-38h]
  size_t v42; // [rsp+8h] [rbp-38h]

  v10 = sub_16E8C20(a1, a2, a3);
  v11 = sub_1263B40(v10, "  -");
  v12 = *(_QWORD *)(a2 + 32);
  v13 = *(const char **)(a2 + 24);
  v14 = *(void **)(v11 + 24);
  if ( v12 > *(_QWORD *)(v11 + 16) - (_QWORD)v14 )
  {
    v14 = (void *)v11;
    sub_16E7EE0(v11, v13);
  }
  else if ( v12 )
  {
    v40 = v11;
    v42 = v12;
    memcpy(v14, v13, v12);
    v12 = v42;
    *(_QWORD *)(v40 + 24) += v42;
  }
  v15 = sub_16E8C20(v14, v13, v12);
  v16 = (unsigned int)(a5 - *(_DWORD *)(a2 + 32));
  sub_16E8750(v15, v16);
  v17 = a1;
  v41 = ((__int64 (__fastcall *)(unsigned __int8 (__fastcall ***)(_QWORD, _QWORD)))(*a1)[2])(a1);
  if ( v41 )
  {
    v19 = 0;
    while ( 1 )
    {
      v20 = **a3;
      v17 = a3;
      v16 = ((__int64 (__fastcall *)(_QWORD, _QWORD))(*a1)[6])(a1, v19);
      if ( !v20(a3, v16) )
        break;
      if ( v41 == ++v19 )
        goto LABEL_8;
    }
    v23 = sub_16E8C20(a3, v16, v18);
    v24 = sub_1263B40(v23, "= ");
    v25 = (const char *)((__int64 (__fastcall *)(_QWORD, _QWORD))(*a1)[3])(a1, v19);
    sub_1549FF0(v24, v25, v26);
    v27 = v19;
    v28 = 0;
    (*a1)[3](a1, v27);
    if ( v29 < 8 )
      v28 = 8 - v29;
    v30 = sub_16E8C20(a1, 8 - v29, v29);
    v31 = v28;
    v32 = 0;
    v33 = sub_16E8750(v30, v31);
    sub_1263B40(v33, " (default: ");
    while ( 1 )
    {
      v34 = **a4;
      v35 = (__int64)a4;
      v36 = (const char *)((__int64 (__fastcall *)(_QWORD, _QWORD))(*a1)[6])(a1, v32);
      if ( !v34(a4, v36) )
        break;
      if ( v41 == ++v32 )
        goto LABEL_14;
    }
    v35 = sub_16E8C20(a4, v36, v37);
    v36 = (const char *)((__int64 (__fastcall *)(_QWORD, _QWORD))(*a1)[3])(a1, v32);
    sub_1549FF0(v35, v36, v39);
LABEL_14:
    v38 = sub_16E8C20(v35, v36, v37);
    return sub_1263B40(v38, ")\n");
  }
  else
  {
LABEL_8:
    v21 = sub_16E8C20(v17, v16, v18);
    return sub_1263B40(v21, "= *unknown option value*\n");
  }
}
