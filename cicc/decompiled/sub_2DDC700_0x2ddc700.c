// Function: sub_2DDC700
// Address: 0x2ddc700
//
void __fastcall sub_2DDC700(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r14
  int v4; // r13d
  const char *v5; // rax
  unsigned __int64 v6; // rdx
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  __int64 (__fastcall *v9)(unsigned __int8 *, unsigned int); // rsi
  __int64 v10; // rdx
  __int64 (__fastcall *v11)(unsigned __int8 *, unsigned int); // rsi
  _QWORD *v12; // rdx
  __int64 v13; // r9
  int v14; // r8d
  unsigned __int64 v15; // r12
  unsigned __int64 v16; // r12
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rax
  const __m128i *v20; // rax
  const __m128i *v21; // r15
  __int32 v22; // edx
  const __m128i *v23; // r13
  unsigned __int64 v24; // rcx
  unsigned int v25; // eax
  __int64 v26; // rdx
  __m128i *v27; // rdx
  __int64 v28; // r9
  __int64 v29; // rax
  __int64 v30; // r8
  __int64 *v31; // rdx
  _BYTE *v32; // rsi
  size_t v33; // rdx
  _BYTE *v34; // rdi
  int v35; // [rsp+4h] [rbp-19Ch]
  int v36; // [rsp+4h] [rbp-19Ch]
  __int64 v39; // [rsp+20h] [rbp-180h]
  __int64 v40; // [rsp+28h] [rbp-178h]
  __int64 v41; // [rsp+38h] [rbp-168h]
  __int64 (__fastcall *v42)(unsigned __int8 *, unsigned int); // [rsp+40h] [rbp-160h] BYREF
  unsigned __int64 v43; // [rsp+48h] [rbp-158h]
  unsigned __int64 v44; // [rsp+50h] [rbp-150h]
  __int64 (__fastcall *v45)(unsigned __int8 *, unsigned int); // [rsp+60h] [rbp-140h] BYREF
  unsigned __int64 v46; // [rsp+68h] [rbp-138h]
  _BYTE v47[16]; // [rsp+70h] [rbp-130h] BYREF
  _BYTE *v48[2]; // [rsp+80h] [rbp-120h] BYREF
  _QWORD v49[2]; // [rsp+90h] [rbp-110h] BYREF
  _BYTE *v50; // [rsp+A0h] [rbp-100h] BYREF
  __int64 v51; // [rsp+A8h] [rbp-F8h]
  _BYTE v52[48]; // [rsp+B0h] [rbp-F0h] BYREF
  __int64 (__fastcall *v53)(unsigned __int8 *, unsigned int); // [rsp+E0h] [rbp-C0h] BYREF
  _QWORD *v54; // [rsp+E8h] [rbp-B8h] BYREF
  __int64 (__fastcall *v55)(__int64 (__fastcall **)(unsigned __int8 *, unsigned int), __int64 (__fastcall **)(unsigned __int8 *, unsigned int), int); // [rsp+F0h] [rbp-B0h]
  _QWORD v56[2]; // [rsp+F8h] [rbp-A8h] BYREF
  _QWORD *v57; // [rsp+108h] [rbp-98h] BYREF
  _QWORD v58[2]; // [rsp+118h] [rbp-88h] BYREF
  int v59; // [rsp+128h] [rbp-78h]
  _BYTE *v60; // [rsp+130h] [rbp-70h] BYREF
  __int64 v61; // [rsp+138h] [rbp-68h]
  _BYTE dest[96]; // [rsp+140h] [rbp-60h] BYREF

  v2 = *(_QWORD *)(a2 + 32);
  v41 = a2 + 24;
  if ( v2 != a2 + 24 )
  {
    while ( 1 )
    {
      v3 = v2 - 56;
      if ( !v2 )
        v3 = 0;
      if ( !(unsigned __int8)sub_2DDC600(v3) )
        goto LABEL_3;
      v53 = sub_2DDBF40;
      v56[0] = sub_2DDB3F0;
      v55 = (__int64 (__fastcall *)(__int64 (__fastcall **)(unsigned __int8 *, unsigned int), __int64 (__fastcall **)(unsigned __int8 *, unsigned int), int))sub_2DDB400;
      sub_3147BA0(&v42, v3, &v53);
      if ( v55 )
        v55(&v53, &v53, 3);
      v50 = v52;
      v51 = 0x300000000LL;
      if ( *(_DWORD *)(v44 + 16) )
      {
        v20 = *(const __m128i **)(v44 + 8);
        v21 = &v20[*(unsigned int *)(v44 + 24)];
        if ( v20 != v21 )
          break;
      }
LABEL_10:
      v4 = *(_DWORD *)(v43 + 40);
      v48[0] = v49;
      sub_2DDB4F0((__int64 *)v48, *(_BYTE **)(a2 + 168), *(_QWORD *)(a2 + 168) + *(_QWORD *)(a2 + 176));
      v5 = sub_BD5D20(v3);
      v46 = v6;
      v45 = (__int64 (__fastcall *)(unsigned __int8 *, unsigned int))v5;
      v7 = sub_C93460((__int64 *)&v45, ".content.", 9u);
      if ( v7 == -1
        || (v8 = v7 + 9, v8 > v46)
        || (v9 = (__int64 (__fastcall *)(unsigned __int8 *, unsigned int))((char *)v45 + v8), v10 = v46 - v8, v46 == v8) )
      {
        v18 = sub_C93460((__int64 *)&v45, ".llvm.", 6u);
        if ( v18 == -1 )
        {
          v18 = v46;
        }
        else if ( v46 <= v18 )
        {
          v18 = v46;
        }
        v53 = v45;
        v54 = (_QWORD *)v18;
        v19 = sub_C93460((__int64 *)&v53, ".__uniq.", 8u);
        v9 = v53;
        v10 = v19;
        if ( v19 == -1 )
        {
          v10 = (__int64)v54;
        }
        else if ( (unsigned __int64)v54 <= v19 )
        {
          v10 = (__int64)v54;
        }
      }
      if ( v9 )
      {
        v45 = (__int64 (__fastcall *)(unsigned __int8 *, unsigned int))v47;
        sub_2DDBAF0((__int64 *)&v45, v9, (__int64)v9 + v10);
        v11 = v45;
        v12 = (_QWORD *)((char *)v45 + v46);
      }
      else
      {
        v12 = v47;
        v47[0] = 0;
        v45 = (__int64 (__fastcall *)(unsigned __int8 *, unsigned int))v47;
        v11 = (__int64 (__fastcall *)(unsigned __int8 *, unsigned int))v47;
        v46 = 0;
      }
      v54 = v56;
      v53 = v42;
      sub_2DDB4F0((__int64 *)&v54, v11, (__int64)v12);
      v57 = v58;
      sub_2DDB4F0((__int64 *)&v57, v48[0], (__int64)&v48[0][(unsigned __int64)v48[1]]);
      v14 = v51;
      v59 = v4;
      v61 = 0x300000000LL;
      v60 = dest;
      if ( !(_DWORD)v51 )
        goto LABEL_16;
      if ( v50 != v52 )
      {
        v60 = v50;
        v61 = v51;
        v51 = 0;
        v50 = v52;
        goto LABEL_16;
      }
      if ( (unsigned int)v51 > 3 )
      {
        v36 = v51;
        sub_C8D5F0((__int64)&v60, dest, (unsigned int)v51, 0x10u, (unsigned int)v51, v13);
        v34 = v60;
        v32 = v50;
        v14 = v36;
        v33 = 16LL * (unsigned int)v51;
        if ( !v33 )
          goto LABEL_69;
      }
      else
      {
        v32 = v52;
        v33 = 16LL * (unsigned int)v51;
        v34 = dest;
      }
      v35 = v14;
      memcpy(v34, v32, v33);
      v14 = v35;
LABEL_69:
      LODWORD(v61) = v14;
      LODWORD(v51) = 0;
LABEL_16:
      if ( (char *)v45 != v47 )
        j_j___libc_free_0((unsigned __int64)v45);
      if ( (_QWORD *)v48[0] != v49 )
        j_j___libc_free_0((unsigned __int64)v48[0]);
      sub_311B700(*(_QWORD *)(a1 + 8), &v53);
      if ( v60 != dest )
        _libc_free((unsigned __int64)v60);
      if ( v57 != v58 )
        j_j___libc_free_0((unsigned __int64)v57);
      if ( v54 != v56 )
        j_j___libc_free_0((unsigned __int64)v54);
      if ( v50 != v52 )
        _libc_free((unsigned __int64)v50);
      v15 = v44;
      if ( v44 )
      {
        sub_C7D6A0(*(_QWORD *)(v44 + 8), 16LL * *(unsigned int *)(v44 + 24), 8);
        j_j___libc_free_0(v15);
      }
      v16 = v43;
      if ( v43 )
      {
        v17 = *(_QWORD *)(v43 + 32);
        if ( v17 != v43 + 48 )
          _libc_free(v17);
        sub_C7D6A0(*(_QWORD *)(v16 + 8), 8LL * *(unsigned int *)(v16 + 24), 4);
        j_j___libc_free_0(v16);
        v2 = *(_QWORD *)(v2 + 8);
        if ( v41 == v2 )
          return;
      }
      else
      {
LABEL_3:
        v2 = *(_QWORD *)(v2 + 8);
        if ( v41 == v2 )
          return;
      }
    }
    v22 = v20->m128i_i32[0];
    v23 = *(const __m128i **)(v44 + 8);
    if ( v20->m128i_i32[0] == -1 )
      goto LABEL_73;
    while ( v22 == -2 && v20->m128i_i32[1] == -2 )
    {
      while ( 1 )
      {
        if ( v21 == ++v20 )
          goto LABEL_10;
        v22 = v20->m128i_i32[0];
        v23 = v20;
        if ( v20->m128i_i32[0] != -1 )
          break;
LABEL_73:
        if ( v20->m128i_i32[1] != -1 )
          goto LABEL_46;
      }
    }
LABEL_46:
    if ( v21 != v20 )
    {
      v24 = 3;
      v25 = 0;
LABEL_51:
      v26 = v25;
      if ( v25 >= v24 )
      {
        v28 = v25 + 1LL;
        v29 = v23->m128i_i64[1];
        v30 = v23->m128i_i64[0];
        if ( v24 < v26 + 1 )
        {
          v39 = v23->m128i_i64[0];
          v40 = v23->m128i_i64[1];
          sub_C8D5F0((__int64)&v50, v52, v26 + 1, 0x10u, v30, v28);
          v26 = (unsigned int)v51;
          v30 = v39;
          v29 = v40;
        }
        v31 = (__int64 *)&v50[16 * v26];
        *v31 = v30;
        v31[1] = v29;
        LODWORD(v51) = v51 + 1;
      }
      else
      {
        v27 = (__m128i *)&v50[16 * v25];
        if ( v27 )
        {
          *v27 = _mm_loadu_si128(v23);
          v25 = v51;
        }
        LODWORD(v51) = v25 + 1;
      }
      if ( ++v23 == v21 )
        goto LABEL_10;
      while ( 1 )
      {
        if ( v23->m128i_i32[0] == -1 )
        {
          if ( v23->m128i_i32[1] != -1 )
            goto LABEL_49;
        }
        else if ( v23->m128i_i32[0] != -2 || v23->m128i_i32[1] != -2 )
        {
LABEL_49:
          if ( v23 == v21 )
            goto LABEL_10;
          v25 = v51;
          v24 = HIDWORD(v51);
          goto LABEL_51;
        }
        if ( v21 == ++v23 )
          goto LABEL_10;
      }
    }
    goto LABEL_10;
  }
}
