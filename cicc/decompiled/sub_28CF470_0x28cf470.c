// Function: sub_28CF470
// Address: 0x28cf470
//
__int64 *__fastcall sub_28CF470(__int64 *a1, __int64 a2, unsigned __int8 *a3)
{
  int v6; // r15d
  __int64 v7; // rax
  int v8; // r15d
  __int64 v9; // rbx
  __int64 v10; // rax
  char v11; // r15
  __int64 *v12; // r8
  bool v13; // al
  __int64 *v14; // r8
  int v15; // ecx
  __int64 *v16; // r8
  unsigned int v17; // r15d
  bool v18; // al
  unsigned int v19; // eax
  unsigned __int8 *v20; // rax
  __int64 v21; // r9
  __int64 *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rsi
  unsigned __int8 v26; // al
  __int64 *v27; // r8
  __int64 *v28; // r10
  unsigned __int64 v29; // rdx
  unsigned int v30; // eax
  __int64 v31; // rcx
  __int64 v32; // r9
  __int64 *v33; // rdx
  __int64 *v34; // rsi
  __int64 v35; // rdx
  unsigned __int8 *v36; // r8
  __int64 v37; // r9
  __int64 *v38; // rdi
  __int64 v39; // [rsp+0h] [rbp-110h]
  __int64 *v40; // [rsp+0h] [rbp-110h]
  __int64 v41; // [rsp+8h] [rbp-108h]
  unsigned __int8 *v42; // [rsp+8h] [rbp-108h]
  __int64 *v43; // [rsp+8h] [rbp-108h]
  unsigned __int8 *v44; // [rsp+10h] [rbp-100h]
  unsigned __int8 v45; // [rsp+10h] [rbp-100h]
  __int64 v46; // [rsp+10h] [rbp-100h]
  __int64 *v47; // [rsp+18h] [rbp-F8h]
  __int64 *v48; // [rsp+18h] [rbp-F8h]
  __int64 v49[4]; // [rsp+20h] [rbp-F0h] BYREF
  __m128i v50[2]; // [rsp+40h] [rbp-D0h] BYREF
  unsigned __int64 v51; // [rsp+60h] [rbp-B0h]
  unsigned __int8 *v52; // [rsp+68h] [rbp-A8h]
  __m128i v53; // [rsp+70h] [rbp-A0h]
  __int64 v54; // [rsp+80h] [rbp-90h]
  __int64 *v55; // [rsp+90h] [rbp-80h] BYREF
  __int64 v56; // [rsp+98h] [rbp-78h]
  _QWORD v57[14]; // [rsp+A0h] [rbp-70h] BYREF

  v6 = *((_DWORD *)a3 + 1);
  v7 = sub_A777F0(0x30u, (__int64 *)(a2 + 72));
  v8 = v6 & 0x7FFFFFF;
  v9 = v7;
  if ( v7 )
  {
    *(_QWORD *)(v7 + 16) = 0;
    *(_QWORD *)(v7 + 24) = 0;
    *(_DWORD *)(v7 + 32) = v8;
    *(_QWORD *)v7 = &unk_4A21A28;
    *(_QWORD *)(v7 + 8) = 0xFFFFFFFD00000006LL;
    *(_DWORD *)(v7 + 36) = 0;
    *(_QWORD *)(v7 + 40) = 0;
  }
  v10 = *(_QWORD *)(a2 + 1344);
  v50[0] = _mm_loadu_si128((const __m128i *)(a2 + 1280));
  v51 = _mm_loadu_si128((const __m128i *)(a2 + 1312)).m128i_u64[0];
  v54 = v10;
  v52 = a3;
  v50[1] = _mm_loadu_si128((const __m128i *)(a2 + 1296));
  v53 = _mm_loadu_si128((const __m128i *)(a2 + 1328));
  v11 = sub_28CF1F0(a2, (__int64 *)a3, v9);
  if ( !sub_B46D50(a3) )
  {
LABEL_6:
    LOBYTE(v15) = *a3;
    if ( (unsigned __int8)(*a3 - 82) <= 1u )
    {
      v16 = *(__int64 **)(v9 + 24);
      v45 = *a3;
      v48 = v16;
      v17 = *((_WORD *)a3 + 1) & 0x3F;
      v39 = v16[1];
      v42 = (unsigned __int8 *)*v16;
      v18 = sub_28C8D50(a2, (unsigned __int8 *)*v16, v39);
      v14 = v48;
      v15 = v45;
      if ( v18 )
      {
        *v48 = v39;
        v48[1] = (__int64)v42;
        v19 = sub_B52F50(v17);
        v14 = *(__int64 **)(v9 + 24);
        v15 = *a3;
        v17 = v19;
      }
      goto LABEL_9;
    }
    goto LABEL_14;
  }
  v12 = *(__int64 **)(v9 + 24);
  v47 = v12;
  v41 = v12[1];
  v44 = (unsigned __int8 *)*v12;
  v13 = sub_28C8D50(a2, (unsigned __int8 *)*v12, v41);
  v14 = v47;
  if ( v13 )
  {
    *v47 = v41;
    v47[1] = (__int64)v44;
    goto LABEL_6;
  }
  v15 = *a3;
  if ( (unsigned __int8)(v15 - 82) <= 1u )
  {
    v17 = *((_WORD *)a3 + 1) & 0x3F;
LABEL_9:
    *(_DWORD *)(v9 + 12) = v17 | ((v15 - 29) << 8);
    v20 = (unsigned __int8 *)sub_10197D0(v17, (_BYTE *)*v14, (_BYTE *)v14[1], v50);
    goto LABEL_10;
  }
LABEL_14:
  if ( (_BYTE)v15 == 86 )
  {
    v23 = *(__int64 **)(v9 + 24);
    v24 = v23[2];
    v25 = v23[1];
    if ( *(_BYTE *)*v23 > 0x15u && v24 != v25 )
      goto LABEL_20;
    v20 = (unsigned __int8 *)sub_1020DD0(*v23, v25, v24, v50);
    goto LABEL_10;
  }
  if ( (unsigned int)(unsigned __int8)v15 - 42 <= 0x11 )
  {
    v20 = sub_101E7C0(*(_DWORD *)(v9 + 12), **(__int64 ***)(v9 + 24), *(__int64 **)(*(_QWORD *)(v9 + 24) + 8LL), v50);
    goto LABEL_10;
  }
  if ( (unsigned int)(unsigned __int8)v15 - 67 <= 0xC )
  {
    v20 = (unsigned __int8 *)sub_1002A60(
                               (unsigned int)(unsigned __int8)v15 - 29,
                               **(unsigned __int8 ***)(v9 + 24),
                               *((_QWORD *)a3 + 1),
                               v50[0].m128i_i64);
    goto LABEL_10;
  }
  if ( (_BYTE)v15 == 63 )
  {
    v26 = sub_B4DE20((__int64)a3);
    v20 = (unsigned __int8 *)sub_100E380(
                               *((_QWORD *)a3 + 9),
                               **(_QWORD **)(v9 + 24),
                               (_QWORD *)(*(_QWORD *)(v9 + 24) + 8LL),
                               (8LL * *(unsigned int *)(v9 + 36) - 8) >> 3,
                               v26,
                               v50[0].m128i_i64);
LABEL_10:
    sub_28CED90((__int64 *)&v55, a2, v9, (__int64)a3, v20, v21);
    if ( v55 )
    {
      *a1 = (__int64)v55;
      a1[1] = v56;
      a1[2] = v57[0];
      return a1;
    }
LABEL_20:
    *a1 = v9;
    a1[1] = 0;
    a1[2] = 0;
    return a1;
  }
  if ( !v11 )
    goto LABEL_20;
  v55 = v57;
  v56 = 0x800000000LL;
  v27 = *(__int64 **)(v9 + 24);
  v28 = &v27[*(unsigned int *)(v9 + 36)];
  if ( v27 == v28 )
  {
    v35 = 0;
    v34 = v57;
  }
  else
  {
    v29 = 8;
    v30 = 0;
    while ( 1 )
    {
      v31 = v30;
      v32 = *v27;
      if ( v30 >= v29 )
      {
        if ( v29 < (unsigned __int64)v30 + 1 )
        {
          v40 = v27;
          v43 = v28;
          v46 = *v27;
          sub_C8D5F0((__int64)&v55, v57, v30 + 1LL, 8u, (__int64)v27, v32);
          v31 = (unsigned int)v56;
          v27 = v40;
          v28 = v43;
          v32 = v46;
        }
        v55[v31] = v32;
        v30 = v56 + 1;
        LODWORD(v56) = v56 + 1;
      }
      else
      {
        v33 = &v55[v30];
        if ( v33 )
        {
          *v33 = v32;
          v30 = v56;
        }
        LODWORD(v56) = ++v30;
      }
      if ( v28 == ++v27 )
        break;
      v29 = HIDWORD(v56);
    }
    v34 = v55;
    v35 = v30;
  }
  v36 = (unsigned __int8 *)sub_97D230(a3, v34, v35, *(_BYTE **)(a2 + 56), *(__int64 **)(a2 + 16), 1u);
  if ( !v36 || (sub_28CED90(v49, a2, v9, (__int64)a3, v36, v37), !v49[0]) )
  {
    if ( v55 != v57 )
      _libc_free((unsigned __int64)v55);
    goto LABEL_20;
  }
  *a1 = v49[0];
  v38 = v55;
  a1[1] = v49[1];
  a1[2] = v49[2];
  if ( v38 != v57 )
    _libc_free((unsigned __int64)v38);
  return a1;
}
