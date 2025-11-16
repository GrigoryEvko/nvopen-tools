// Function: sub_E55510
// Address: 0xe55510
//
char *__fastcall sub_E55510(
        __int64 a1,
        unsigned int a2,
        char *a3,
        __int64 a4,
        const char *a5,
        size_t a6,
        char a7,
        int a8,
        char a9,
        char *a10,
        __int64 a11,
        char a12,
        char a13,
        __int64 a14)
{
  unsigned int v14; // r10d
  char *v15; // r15
  __int64 v16; // r14
  __int64 v18; // rbx
  __int64 v19; // r12
  __int64 v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // rdi
  _BYTE *v23; // rax
  char *v24; // rsi
  char *result; // rax
  __int64 v26; // r14
  _BYTE *v27; // rax
  __int64 v28; // rax
  char v29; // al
  unsigned int v32; // [rsp+28h] [rbp-1B8h]
  char *v33; // [rsp+30h] [rbp-1B0h]
  _QWORD v34[4]; // [rsp+40h] [rbp-1A0h] BYREF
  __int16 v35; // [rsp+60h] [rbp-180h]
  _BYTE v36[32]; // [rsp+70h] [rbp-170h] BYREF
  __int16 v37; // [rsp+90h] [rbp-150h]
  _BYTE v38[32]; // [rsp+A0h] [rbp-140h] BYREF
  __int16 v39; // [rsp+C0h] [rbp-120h]
  char *v40; // [rsp+D0h] [rbp-110h] BYREF
  size_t v41; // [rsp+D8h] [rbp-108h]
  char v42; // [rsp+E8h] [rbp-F8h] BYREF
  __int16 v43; // [rsp+F0h] [rbp-F0h]
  char *v44; // [rsp+110h] [rbp-D0h] BYREF
  __int64 v45; // [rsp+118h] [rbp-C8h]
  __int64 v46; // [rsp+120h] [rbp-C0h]
  _BYTE v47[184]; // [rsp+128h] [rbp-B8h] BYREF

  v14 = a2;
  v15 = (char *)a5;
  v16 = a6;
  v18 = a4;
  v33 = a3;
  v19 = a14;
  v44 = v47;
  v45 = 0;
  v46 = 128;
  if ( !a13 )
  {
    if ( a4 )
    {
      v18 = 0;
      v43 = 261;
      v40 = (char *)a5;
      v41 = a6;
      v29 = sub_C81DB0((const char **)&v40, 0);
      v14 = a2;
      v33 = (char *)byte_3F871B3;
      if ( !v29 )
      {
        v45 = 0;
        sub_C58CA0(&v44, a3, &a3[a4]);
        v43 = 257;
        v39 = 257;
        v37 = 257;
        v35 = 261;
        v34[0] = v15;
        v34[1] = v16;
        sub_C81B70(&v44, (__int64)v34, (__int64)v36, (__int64)v38, (__int64)&v40);
        v16 = v45;
        v15 = v44;
        v14 = a2;
      }
    }
  }
  v20 = *(_QWORD *)(v19 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v19 + 24) - v20) <= 6 )
  {
    v32 = v14;
    v28 = sub_CB6200(v19, "\t.file\t", 7u);
    v14 = v32;
    v21 = v28;
  }
  else
  {
    *(_DWORD *)v20 = 1768304137;
    v21 = v19;
    *(_WORD *)(v20 + 4) = 25964;
    *(_BYTE *)(v20 + 6) = 9;
    *(_QWORD *)(v19 + 32) += 7LL;
  }
  v22 = sub_CB59D0(v21, v14);
  v23 = *(_BYTE **)(v22 + 32);
  if ( (unsigned __int64)v23 >= *(_QWORD *)(v22 + 24) )
  {
    sub_CB5D20(v22, 32);
  }
  else
  {
    *(_QWORD *)(v22 + 32) = v23 + 1;
    *v23 = 32;
  }
  if ( v18 )
  {
    sub_E51560(a1, v33, v18, v19);
    v27 = *(_BYTE **)(v19 + 32);
    if ( (unsigned __int64)v27 >= *(_QWORD *)(v19 + 24) )
    {
      sub_CB5D20(v19, 32);
    }
    else
    {
      *(_QWORD *)(v19 + 32) = v27 + 1;
      *v27 = 32;
    }
  }
  v24 = v15;
  result = (char *)sub_E51560(a1, v15, v16, v19);
  if ( a9 )
  {
    v26 = sub_904010(v19, " md5 0x");
    sub_C7D470(&v40, (unsigned __int8 *)&a7);
    v24 = v40;
    sub_CB6200(v26, (unsigned __int8 *)v40, v41);
    result = &v42;
    if ( v40 != &v42 )
      result = (char *)_libc_free(v40, v24);
  }
  if ( a12 )
  {
    sub_904010(v19, " source ");
    v24 = a10;
    result = (char *)sub_E51560(a1, a10, a11, v19);
  }
  if ( v44 != v47 )
    return (char *)_libc_free(v44, v24);
  return result;
}
