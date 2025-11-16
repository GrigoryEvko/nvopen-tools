// Function: sub_3870D00
// Address: 0x3870d00
//
void __fastcall sub_3870D00(__int64 a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  __int64 v7; // r9
  char *v8; // r8
  char *v9; // rdx
  unsigned int v10; // eax
  __int64 v11; // rbx
  unsigned __int64 v12; // r9
  size_t v13; // r10
  _BYTE *v14; // rdi
  __int64 v15; // rax
  __int64 v16; // r10
  __int64 v17; // rax
  char *v18; // r9
  char *v19; // rcx
  signed __int64 v20; // r10
  char *v21; // r8
  int v22; // eax
  __int64 v23; // r11
  __int64 v24; // rdi
  bool v25; // zf
  int v26; // r9d
  void *v27; // r8
  size_t v28; // r13
  int v29; // r12d
  __int64 v30; // rdx
  _BYTE *v31; // rdi
  signed __int64 v32; // [rsp+0h] [rbp-100h]
  char *src; // [rsp+8h] [rbp-F8h]
  char *srca; // [rsp+8h] [rbp-F8h]
  size_t n; // [rsp+10h] [rbp-F0h]
  size_t na; // [rsp+10h] [rbp-F0h]
  int v37; // [rsp+18h] [rbp-E8h]
  int v38; // [rsp+18h] [rbp-E8h]
  int v39; // [rsp+18h] [rbp-E8h]
  void *v40; // [rsp+18h] [rbp-E8h]
  __int64 v41; // [rsp+18h] [rbp-E8h]
  __int64 v42; // [rsp+28h] [rbp-D8h] BYREF
  _BYTE *v43; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v44; // [rsp+38h] [rbp-C8h]
  _BYTE v45[64]; // [rsp+40h] [rbp-C0h] BYREF
  void *v46; // [rsp+80h] [rbp-80h] BYREF
  __int64 v47; // [rsp+88h] [rbp-78h]
  _BYTE dest[112]; // [rsp+90h] [rbp-70h] BYREF

  v7 = *(unsigned int *)(a1 + 8);
  v8 = *(char **)a1;
  if ( (_DWORD)v7 )
  {
    v9 = &v8[8 * (unsigned int)(v7 - 1)];
    v10 = 0;
    do
    {
      if ( *(_WORD *)(*(_QWORD *)v9 + 24LL) != 7 )
      {
        v11 = v10;
        v12 = v7 - v10;
        v43 = v45;
        v13 = 8 * v12;
        v44 = 0x800000000LL;
        if ( 8 * v12 <= 0x40 )
          goto LABEL_26;
        src = v8;
        n = 8 * v12;
        v37 = v12;
        sub_16CD150((__int64)&v43, v45, v12, 8, (int)v8, v12);
        LODWORD(v12) = v37;
        v13 = n;
        v8 = src;
        v14 = &v43[8 * (unsigned int)v44];
LABEL_7:
        v38 = v12;
        memcpy(v14, v8, v13);
        LODWORD(v13) = v44;
        v8 = *(char **)a1;
        LODWORD(v12) = v38;
        goto LABEL_8;
      }
      ++v10;
      v9 -= 8;
    }
    while ( v10 != (_DWORD)v7 );
    v11 = v10;
    v43 = v45;
    v12 = v7 - v10;
    v44 = 0x800000000LL;
    v13 = 8 * v12;
LABEL_26:
    if ( v13 )
    {
      v14 = v45;
      goto LABEL_7;
    }
  }
  else
  {
    LODWORD(v12) = 0;
    v11 = 0;
    LODWORD(v13) = 0;
    v43 = v45;
    HIDWORD(v44) = 8;
  }
LABEL_8:
  v15 = *(unsigned int *)(a1 + 8);
  LODWORD(v44) = v13 + v12;
  v16 = 8 * v15;
  v47 = 0x800000000LL;
  v17 = 8 * (v15 - v11);
  v18 = &v8[v16];
  v46 = dest;
  v19 = dest;
  v20 = v16 - v17;
  v21 = &v8[v17];
  v22 = 0;
  v23 = v20 >> 3;
  if ( (unsigned __int64)v20 > 0x40 )
  {
    v32 = v20;
    srca = v18;
    na = (size_t)v21;
    v41 = v20 >> 3;
    sub_16CD150((__int64)&v46, dest, v20 >> 3, 8, (int)v21, (int)v18);
    v22 = v47;
    v20 = v32;
    v18 = srca;
    v21 = (char *)na;
    LODWORD(v23) = v41;
    v19 = (char *)v46 + 8 * (unsigned int)v47;
  }
  if ( v21 != v18 )
  {
    v39 = v23;
    memcpy(v19, v21, v20);
    v22 = v47;
    LODWORD(v23) = v39;
  }
  LODWORD(v47) = v22 + v23;
  if ( (_DWORD)v44 )
    v24 = (__int64)sub_147DD40(a3, (__int64 *)&v43, 0, 0, a4, a5);
  else
    v24 = sub_145CF80(a3, a2, 0, 0);
  *(_DWORD *)(a1 + 8) = 0;
  v25 = *(_WORD *)(v24 + 24) == 4;
  v42 = v24;
  if ( v25 )
  {
    sub_145C5B0(a1, *(_BYTE **)(v24 + 32), (_BYTE *)(*(_QWORD *)(v24 + 32) + 8LL * *(_QWORD *)(v24 + 40)));
  }
  else if ( !sub_14560B0(v24) )
  {
    sub_1458920(a1, &v42);
  }
  v27 = v46;
  v28 = 8LL * (unsigned int)v47;
  v29 = v47;
  v30 = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)v47 > (unsigned __int64)*(unsigned int *)(a1 + 12) - v30 )
  {
    v40 = v46;
    sub_16CD150(a1, (const void *)(a1 + 16), (unsigned int)v47 + v30, 8, (int)v46, v26);
    v30 = *(unsigned int *)(a1 + 8);
    v27 = v40;
  }
  if ( v28 )
  {
    memcpy((void *)(*(_QWORD *)a1 + 8 * v30), v27, v28);
    LODWORD(v30) = *(_DWORD *)(a1 + 8);
  }
  v31 = v46;
  *(_DWORD *)(a1 + 8) = v29 + v30;
  if ( v31 != dest )
    _libc_free((unsigned __int64)v31);
  if ( v43 != v45 )
    _libc_free((unsigned __int64)v43);
}
