// Function: sub_3022E70
// Address: 0x3022e70
//
void __fastcall sub_3022E70(__int64 a1, __int64 a2, __int64 a3)
{
  char v5; // bl
  __int64 v6; // rax
  __m128i *v7; // rdx
  __m128i si128; // xmm0
  __int64 v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // rax
  void *v12; // rdx
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  char *v17; // [rsp+0h] [rbp-D0h] BYREF
  unsigned int v18; // [rsp+8h] [rbp-C8h]
  char v19; // [rsp+10h] [rbp-C0h] BYREF
  char *v20; // [rsp+20h] [rbp-B0h] BYREF
  int v21; // [rsp+28h] [rbp-A8h]
  char v22; // [rsp+30h] [rbp-A0h] BYREF
  char *v23; // [rsp+40h] [rbp-90h] BYREF
  int v24; // [rsp+48h] [rbp-88h]
  char v25; // [rsp+50h] [rbp-80h] BYREF
  const char *v26; // [rsp+60h] [rbp-70h] BYREF
  __int64 v27; // [rsp+68h] [rbp-68h]
  _QWORD *v28; // [rsp+70h] [rbp-60h]
  __int64 v29; // [rsp+78h] [rbp-58h]
  char v30; // [rsp+80h] [rbp-50h]
  void *v31; // [rsp+88h] [rbp-48h] BYREF
  char *v32; // [rsp+90h] [rbp-40h]
  char *v33; // [rsp+98h] [rbp-38h]
  _QWORD v34[6]; // [rsp+A0h] [rbp-30h] BYREF

  sub_CE8DF0((__int64)&v17, a2);
  v5 = sub_B2D620(a2, "nvvm.blocksareclusters", 0x16u);
  if ( v5 )
  {
    v7 = *(__m128i **)(a3 + 32);
    if ( *(_QWORD *)(a3 + 24) - (_QWORD)v7 <= 0x12u )
    {
      sub_CB6200(a3, ".blocksareclusters\n", 0x13u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_44C2D70);
      v7[1].m128i_i8[2] = 10;
      v7[1].m128i_i16[0] = 29554;
      *v7 = si128;
      *(_QWORD *)(a3 + 32) += 19LL;
    }
    v6 = v18;
    if ( !v18 )
      sub_C64ED0("blocksareclusters requires reqntid", 1u);
  }
  else
  {
    v6 = v18;
    if ( !v18 )
      goto LABEL_3;
  }
  v30 = 1;
  v26 = ".reqntid {0:$[, ]}\n";
  v28 = v34;
  v27 = 19;
  v33 = &v17[4 * v6];
  v29 = 1;
  v31 = &unk_4A2E380;
  v32 = v17;
  v34[0] = &v31;
  sub_CB6840(a3, (__int64)&v26);
LABEL_3:
  sub_CE8D40((__int64)&v20, a2);
  if ( v21 )
  {
    v30 = 1;
    v26 = ".maxntid {0:$[, ]}\n";
    v28 = v34;
    v27 = 19;
    v33 = &v20[4 * v21];
    v29 = 1;
    v31 = &unk_4A2E380;
    v32 = v20;
    v34[0] = &v31;
    sub_CB6840(a3, (__int64)&v26);
  }
  v26 = (const char *)sub_CE90E0(a2);
  if ( BYTE4(v26) )
  {
    v12 = *(void **)(a3 + 32);
    if ( *(_QWORD *)(a3 + 24) - (_QWORD)v12 <= 0xDu )
    {
      v13 = sub_CB6200(a3, ".minnctapersm ", 0xEu);
    }
    else
    {
      v13 = a3;
      qmemcpy(v12, ".minnctapersm ", 14);
      *(_QWORD *)(a3 + 32) += 14LL;
    }
    v14 = sub_CB59D0(v13, (unsigned int)v26);
    sub_904010(v14, "\n");
  }
  v26 = (const char *)sub_CE9180(a2);
  if ( BYTE4(v26) )
  {
    v9 = *(_QWORD *)(a3 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a3 + 24) - v9) <= 8 )
    {
      v10 = sub_CB6200(a3, ".maxnreg ", 9u);
    }
    else
    {
      *(_BYTE *)(v9 + 8) = 32;
      v10 = a3;
      *(_QWORD *)v9 = 0x6765726E78616D2ELL;
      *(_QWORD *)(a3 + 32) += 9LL;
    }
    v11 = sub_CB59D0(v10, (unsigned int)v26);
    sub_904010(v11, "\n");
  }
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 200) + 1628LL) > 0x383u )
  {
    sub_CE8EA0((__int64)&v23, a2);
    if ( v24 )
    {
      if ( !v5 )
        sub_904010(a3, ".explicitcluster\n");
      if ( *(_DWORD *)v23 )
      {
        v32 = v23;
        v28 = v34;
        v26 = ".reqnctapercluster {0:$[, ]}\n";
        v27 = 29;
        v29 = 1;
        v30 = 1;
        v31 = &unk_4A2E380;
        v33 = &v23[4 * v24];
        v34[0] = &v31;
        sub_CB6840(a3, (__int64)&v26);
      }
    }
    v26 = (const char *)sub_CE9030(a2);
    if ( BYTE4(v26) )
    {
      v15 = sub_904010(a3, ".maxclusterrank ");
      v16 = sub_CB59D0(v15, (unsigned int)v26);
      sub_904010(v16, "\n");
    }
    if ( v23 != &v25 )
      _libc_free((unsigned __int64)v23);
  }
  if ( v20 != &v22 )
    _libc_free((unsigned __int64)v20);
  if ( v17 != &v19 )
    _libc_free((unsigned __int64)v17);
}
