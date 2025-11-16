// Function: sub_2451540
// Address: 0x2451540
//
__int64 __fastcall sub_2451540(_QWORD **a1, __int64 a2, int a3)
{
  __int64 v3; // r14
  unsigned __int8 v5; // dl
  char v6; // al
  int v7; // edx
  int v8; // eax
  char v9; // cl
  int v10; // ebx
  _QWORD *v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rax
  _QWORD *v15; // rsi
  __int64 *v16; // rax
  unsigned __int64 v17; // rsi
  __int64 v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // r14
  char v21; // al
  size_t v23; // rdx
  _QWORD *v24; // rdi
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // r8
  __int64 v28; // rax
  _QWORD *v29; // rdi
  __int64 *v30; // rax
  __int64 v31; // rax
  _QWORD *v32; // rax
  size_t v33; // rdx
  __int64 *v34; // r14
  __int64 v35; // rax
  __int64 v36; // r8
  __int64 v37; // r14
  __int64 *v38; // rax
  __int64 *v39; // rcx
  __int64 *v40; // r10
  __int64 v41; // rax
  _QWORD *v42; // rax
  unsigned __int64 v43; // r10
  __int64 v44; // rax
  unsigned int v45; // eax
  __int64 *v46; // [rsp+10h] [rbp-D0h]
  size_t v47; // [rsp+18h] [rbp-C8h]
  _QWORD *v48; // [rsp+18h] [rbp-C8h]
  unsigned __int64 v49; // [rsp+18h] [rbp-C8h]
  __int64 v50; // [rsp+18h] [rbp-C8h]
  __int64 v51; // [rsp+18h] [rbp-C8h]
  _QWORD *v52; // [rsp+20h] [rbp-C0h]
  __int64 v53; // [rsp+20h] [rbp-C0h]
  __int64 v54; // [rsp+20h] [rbp-C0h]
  __int64 **v55; // [rsp+20h] [rbp-C0h]
  _QWORD *v56; // [rsp+28h] [rbp-B8h]
  size_t v57; // [rsp+28h] [rbp-B8h]
  __int64 v58; // [rsp+30h] [rbp-B0h]
  _QWORD *v59; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v60; // [rsp+30h] [rbp-B0h]
  __int64 v61; // [rsp+38h] [rbp-A8h]
  char v62; // [rsp+48h] [rbp-98h]
  char v64; // [rsp+57h] [rbp-89h] BYREF
  __int64 v65; // [rsp+58h] [rbp-88h]
  void *dest; // [rsp+60h] [rbp-80h]
  size_t v67; // [rsp+68h] [rbp-78h]
  _QWORD v68[2]; // [rsp+70h] [rbp-70h] BYREF
  _QWORD *v69; // [rsp+80h] [rbp-60h] BYREF
  size_t n; // [rsp+88h] [rbp-58h]
  _QWORD src[2]; // [rsp+90h] [rbp-50h] BYREF
  __int16 v72; // [rsp+A0h] [rbp-40h]

  v3 = a2;
  v5 = sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), a2)[32];
  v6 = (v5 >> 4) & 3;
  v7 = v5 & 0xF;
  v61 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL);
  v62 = v6;
  v8 = *((_DWORD *)a1 + 25);
  if ( !unk_4FE76C8 && unk_4FE7468 != 1 || v8 != 5 || (v10 = 7, v7 != 8) )
  {
    v9 = 0;
    if ( v8 != 8 )
    {
      v9 = v62;
      v8 = v7;
    }
    v62 = v9;
    v10 = v8;
  }
  LOBYTE(v68[0]) = 0;
  dest = v68;
  v67 = 0;
  if ( a3 != 1 )
  {
    sub_2451270((__int64 *)&v69, a2, "__profbm_", (void *)9, &v64);
    v11 = dest;
    if ( v69 == src )
    {
      v23 = n;
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)dest = src[0];
        else
          memcpy(dest, src, n);
        v23 = n;
        v11 = dest;
      }
      v67 = v23;
      *((_BYTE *)v11 + v23) = 0;
      v11 = v69;
    }
    else
    {
      if ( dest == v68 )
      {
        dest = v69;
        v67 = n;
        v68[0] = src[0];
      }
      else
      {
        v12 = v68[0];
        dest = v69;
        v67 = n;
        v68[0] = src[0];
        if ( v11 )
        {
          v69 = v11;
          src[0] = v12;
          goto LABEL_13;
        }
      }
      v69 = src;
      v11 = src;
    }
LABEL_13:
    n = 0;
    *(_BYTE *)v11 = 0;
    if ( v69 != src )
      j_j___libc_free_0((unsigned __int64)v69);
    v13 = *(_QWORD *)(v3 - 32);
    if ( v13 && !*(_BYTE *)v13 && *(_QWORD *)(v13 + 24) == *(_QWORD *)(v3 + 80) )
    {
      if ( (unsigned int)(*(_DWORD *)(v13 + 36) - 200) > 1 )
        BUG();
      v52 = dest;
      v47 = v67;
      v14 = *(_QWORD *)(v3 + 32 * (2LL - (*(_DWORD *)(v3 + 4) & 0x7FFFFFF)));
      v15 = *(_QWORD **)(v14 + 24);
      if ( *(_DWORD *)(v14 + 32) > 0x40u )
        v15 = (_QWORD *)*v15;
      v16 = (__int64 *)sub_BCB2B0((_QWORD *)**a1);
      v17 = ((v15 != 0) + (((unsigned __int64)v15 - (v15 != 0)) >> 3)) & 0x1FFFFFFFFFFFFFFFLL;
      v56 = sub_BCD420(v16, v17);
      v18 = sub_AD6530((__int64)v56, v17);
      BYTE4(v65) = 0;
      v58 = v18;
      v72 = 261;
      v69 = v52;
      n = v47;
      v19 = sub_BD2C40(88, unk_3F0FAE8);
      v20 = (__int64)v19;
      if ( v19 )
        sub_B30000((__int64)v19, (__int64)*a1, v56, 0, v10, v58, (__int64)&v69, 0, 0, v65, 0);
      sub_B2F770(v20, 0);
      goto LABEL_24;
    }
LABEL_100:
    BUG();
  }
  sub_2451270((__int64 *)&v69, a2, "__profc_", (void *)8, &v64);
  v24 = dest;
  if ( v69 == src )
  {
    v33 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v33 = n;
      v24 = dest;
    }
    v67 = v33;
    *((_BYTE *)v24 + v33) = 0;
    v24 = v69;
  }
  else
  {
    if ( dest == v68 )
    {
      dest = v69;
      v67 = n;
      v68[0] = src[0];
    }
    else
    {
      v25 = v68[0];
      dest = v69;
      v67 = n;
      v68[0] = src[0];
      if ( v24 )
      {
        v69 = v24;
        src[0] = v25;
        goto LABEL_51;
      }
    }
    v69 = src;
    v24 = src;
  }
LABEL_51:
  n = 0;
  *(_BYTE *)v24 = 0;
  if ( v69 != src )
    j_j___libc_free_0((unsigned __int64)v69);
  if ( *(_BYTE *)v3 == 85
    && (v44 = *(_QWORD *)(v3 - 32)) != 0
    && !*(_BYTE *)v44
    && *(_QWORD *)(v44 + 24) == *(_QWORD *)(v3 + 80)
    && (*(_BYTE *)(v44 + 33) & 0x20) != 0
    && v3 )
  {
    v45 = *(_DWORD *)(v44 + 36);
    if ( v45 > 0xC7 )
    {
      if ( v45 - 202 >= 2 )
        v3 = 0;
    }
    else if ( v45 <= 0xC3 )
    {
      v3 = 0;
    }
  }
  else
  {
    v3 = 0;
  }
  v59 = dest;
  v57 = v67;
  v26 = sub_B59B70(v3);
  if ( *(_DWORD *)(v26 + 32) <= 0x40u )
    v27 = *(_QWORD *)(v26 + 24);
  else
    v27 = **(_QWORD **)(v26 + 24);
  v28 = *(_QWORD *)(v3 - 32);
  if ( !v28 || *(_BYTE *)v28 || *(_QWORD *)(v28 + 24) != *(_QWORD *)(v3 + 80) )
    goto LABEL_100;
  v53 = v27;
  v29 = (_QWORD *)**a1;
  if ( *(_DWORD *)(v28 + 36) == 197 )
  {
    v34 = (__int64 *)sub_BCB2B0(v29);
    v49 = v53;
    v55 = (__int64 **)sub_BCD420(v34, v53);
    v35 = sub_AD62B0((__int64)v34);
    v36 = v49;
    if ( v49 > 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v50 = v35;
    v37 = v36;
    if ( v36 )
    {
      v38 = (__int64 *)sub_22077B0(8 * v36);
      v39 = &v38[v37];
      v40 = v38;
      if ( v38 == &v38[v37] )
      {
        v36 = 0;
      }
      else
      {
        do
          *v38++ = v50;
        while ( v38 != v39 );
        v36 = (v37 * 8) >> 3;
      }
    }
    else
    {
      v40 = 0;
    }
    v46 = v40;
    v41 = sub_AD1300(v55, v40, v36);
    BYTE4(v65) = 0;
    v51 = v41;
    v72 = 261;
    v69 = v59;
    n = v57;
    v42 = sub_BD2C40(88, unk_3F0FAE8);
    v43 = (unsigned __int64)v46;
    v20 = (__int64)v42;
    if ( v42 )
    {
      sub_B30000((__int64)v42, (__int64)*a1, v55, 0, v10, v51, (__int64)&v69, 0, 0, v65, 0);
      v43 = (unsigned __int64)v46;
    }
    v60 = v43;
    sub_B2F770(v20, 0);
    if ( v60 )
      j_j___libc_free_0(v60);
  }
  else
  {
    v30 = (__int64 *)sub_BCB2E0(v29);
    v48 = sub_BCD420(v30, v53);
    v31 = sub_AD6530((__int64)v48, v53);
    BYTE4(v65) = 0;
    v54 = v31;
    v72 = 261;
    v69 = v59;
    n = v57;
    v32 = sub_BD2C40(88, unk_3F0FAE8);
    v20 = (__int64)v32;
    if ( v32 )
      sub_B30000((__int64)v32, (__int64)*a1, v48, 0, v10, v54, (__int64)&v69, 0, 0, v65, 0);
    sub_B2F770(v20, 3u);
  }
LABEL_24:
  v21 = (16 * (v62 & 3)) | *(_BYTE *)(v20 + 32) & 0xCF;
  *(_BYTE *)(v20 + 32) = v21;
  if ( (v21 & 0xFu) - 7 <= 1 || (v21 & 0x30) != 0 && (v21 & 0xF) != 9 )
    *(_BYTE *)(v20 + 33) |= 0x40u;
  sub_ED12E0((__int64)&v69, a3, *((_DWORD *)a1 + 25), 1u);
  sub_B31A00(v20, (__int64)v69, n);
  if ( v69 != src )
    j_j___libc_free_0((unsigned __int64)v69);
  if ( (unsigned int)(v10 - 7) > 1 )
  {
    *(_BYTE *)(v20 + 32) = v10 & 0xF | *(_BYTE *)(v20 + 32) & 0xF0;
  }
  else
  {
    *(_WORD *)(v20 + 32) = v10 & 0x33F | *(_WORD *)(v20 + 32) & 0xFCC0;
    if ( v10 == 7 )
    {
LABEL_30:
      *(_BYTE *)(v20 + 33) |= 0x40u;
      goto LABEL_31;
    }
  }
  if ( v10 == 8 || (*(_BYTE *)(v20 + 32) & 0x30) != 0 && v10 != 9 )
    goto LABEL_30;
LABEL_31:
  sub_24511A0((__int64)a1, v20, v61, dest, v67);
  if ( dest != v68 )
    j_j___libc_free_0((unsigned __int64)dest);
  return v20;
}
