// Function: sub_C533F0
// Address: 0xc533f0
//
__int64 __fastcall sub_C533F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        size_t a5,
        int a6,
        __int64 a7,
        _DWORD *a8)
{
  char *v10; // r8
  __int64 v12; // r12
  int v13; // ebx
  size_t v14; // r9
  unsigned __int8 v15; // al
  int v16; // eax
  __int64 v17; // rsi
  __int64 i; // rdx
  size_t v19; // rax
  char v20; // al
  __int64 v22; // rbx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // r8
  const char *v29; // rax
  size_t v30; // rax
  unsigned __int8 v31; // [rsp-10h] [rbp-160h]
  __int64 v32; // [rsp-8h] [rbp-158h]
  size_t v33; // [rsp+0h] [rbp-150h]
  char v35; // [rsp+18h] [rbp-138h]
  __int64 v36; // [rsp+18h] [rbp-138h]
  char *v37; // [rsp+20h] [rbp-130h]
  __int64 v38; // [rsp+20h] [rbp-130h]
  const char *v39; // [rsp+28h] [rbp-128h]
  size_t v40; // [rsp+28h] [rbp-128h]
  __int64 v41; // [rsp+28h] [rbp-128h]
  const char *v42; // [rsp+28h] [rbp-128h]
  __m128i v43; // [rsp+30h] [rbp-120h] BYREF
  __int16 v44; // [rsp+50h] [rbp-100h]
  __m128i v45[2]; // [rsp+60h] [rbp-F0h] BYREF
  char v46; // [rsp+80h] [rbp-D0h]
  char v47; // [rsp+81h] [rbp-CFh]
  __m128i v48[3]; // [rsp+90h] [rbp-C0h] BYREF
  __m128i v49[2]; // [rsp+C0h] [rbp-90h] BYREF
  char v50; // [rsp+E0h] [rbp-70h]
  char v51; // [rsp+E1h] [rbp-6Fh]
  __m128i v52[2]; // [rsp+F0h] [rbp-60h] BYREF
  char v53; // [rsp+110h] [rbp-40h]
  char v54; // [rsp+111h] [rbp-3Fh]

  v10 = (char *)a4;
  v12 = a1;
  v13 = *(unsigned __int16 *)(a1 + 16);
  v14 = a5;
  v15 = *(_BYTE *)(a1 + 12);
  if ( (v15 & 0x18) == 0 )
  {
    v33 = a5;
    v36 = a4;
    v37 = (char *)a4;
    v40 = a5;
    v16 = (*(__int64 (__fastcall **)(__int64, __int64, size_t, __int64, __int64, size_t))(*(_QWORD *)a1 + 8LL))(
            a1,
            a2,
            a5,
            a4,
            a4,
            a5);
    a5 = v33;
    a4 = v36;
    v10 = v37;
    v14 = v40;
    if ( v16 != 2 )
      goto LABEL_3;
LABEL_16:
    LODWORD(v17) = *a8;
    if ( !a4 )
    {
      v17 = (unsigned int)(v17 + 1);
      if ( (int)v17 >= a6 || ((*(_WORD *)(a1 + 12) ^ 0x180) & 0x180) == 0 )
      {
        v54 = 1;
        v28 = sub_CEADF0(a1, v17, a5, 0, v10, v14);
        v29 = "requires a value!";
        goto LABEL_26;
      }
      a4 = a7;
      *a8 = v17;
      v10 = *(char **)(a7 + 8LL * (int)v17);
      if ( !v10 )
      {
        if ( v13 )
        {
LABEL_6:
          for ( i = 0; ; i = 1 )
          {
            v17 = (unsigned int)(v17 + 1);
            if ( (int)v17 >= a6 )
              break;
            *a8 = v17;
            v14 = 0;
            v10 = *(char **)(a7 + 8LL * (int)v17);
            if ( v10 )
            {
              v35 = i;
              v39 = *(const char **)(a7 + 8LL * (int)v17);
              v19 = strlen(v39);
              LOBYTE(i) = v35;
              v10 = (char *)v39;
              v14 = v19;
            }
            v31 = i;
LABEL_13:
            a1 = v12;
            v20 = sub_C4FF00(v12, v17, a2, a3, v10, v14, v31);
            a4 = v32;
            if ( v20 )
              return 1;
            if ( !--v13 )
              return 0;
            LODWORD(v17) = *a8;
          }
          v54 = 1;
          v28 = sub_CEADF0(a1, v17, i, a4, v10, v14);
          v29 = "not enough values!";
          goto LABEL_26;
        }
        v14 = 0;
        return sub_C4FF00(a1, v17, a2, a3, v10, v14, 0);
      }
      v42 = *(const char **)(a7 + 8LL * (int)v17);
      v30 = strlen(v42);
      v10 = (char *)v42;
      v14 = v30;
    }
    if ( v13 )
      goto LABEL_18;
    return sub_C4FF00(a1, v17, a2, a3, v10, v14, 0);
  }
  v16 = (v15 >> 3) & 3;
  if ( v16 == 2 )
    goto LABEL_16;
LABEL_3:
  if ( v16 != 3 )
  {
    LODWORD(v17) = *a8;
    if ( v13 )
    {
      if ( !a4 )
        goto LABEL_6;
LABEL_18:
      v31 = 0;
      goto LABEL_13;
    }
    return sub_C4FF00(a1, v17, a2, a3, v10, v14, 0);
  }
  if ( !v13 )
  {
    if ( a4 )
    {
      v38 = a5;
      v41 = a4;
      v22 = sub_CEADF0(a1, a2, a5, a4, v10, v14);
      v51 = 1;
      v49[0].m128i_i64[0] = (__int64)"' specified.";
      v44 = 261;
      v43.m128i_i64[1] = v38;
      v43.m128i_i64[0] = v41;
      v45[0].m128i_i64[0] = (__int64)"does not allow a value! '";
      v50 = 3;
      v47 = 1;
      v46 = 3;
      sub_9C6370(v48, v45, &v43, v41, v23, v24);
      sub_9C6370(v52, v48, v49, v25, v26, v27);
      return sub_C53280(a1, (__int64)v52, 0, 0, v22);
    }
    LODWORD(v17) = *a8;
    return sub_C4FF00(a1, v17, a2, a3, v10, v14, 0);
  }
  v54 = 1;
  v28 = sub_CEADF0(a1, a2, a5, a4, v10, v14);
  v29 = "multi-valued option specified with ValueDisallowed modifier!";
LABEL_26:
  v52[0].m128i_i64[0] = (__int64)v29;
  v53 = 3;
  return sub_C53280(v12, (__int64)v52, 0, 0, v28);
}
