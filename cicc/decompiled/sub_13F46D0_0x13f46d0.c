// Function: sub_13F46D0
// Address: 0x13f46d0
//
void __fastcall sub_13F46D0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        __int64 a6,
        unsigned int a7)
{
  __int64 v11; // rax
  unsigned int v12; // r8d
  __int64 v13; // rsi
  char v14; // dl
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r13
  unsigned __int8 v18; // al
  unsigned int v19; // r13d
  unsigned __int64 v20; // rax
  const char *v21; // rax
  _BYTE *v22; // rax
  _BYTE *v23; // rax
  unsigned int v24; // r9d
  __int64 v25; // rdi
  int v26; // eax
  bool v27; // al
  int v28; // eax
  bool v29; // al
  char v30; // al
  bool v31; // r8
  unsigned __int64 v32; // r9
  unsigned __int64 v33; // rax
  char v34; // al
  __int64 v35; // rdx
  unsigned __int64 v36; // rax
  char v37; // al
  unsigned __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rcx
  unsigned int v41; // eax
  unsigned int v42; // [rsp+0h] [rbp-C0h]
  unsigned __int64 v43; // [rsp+8h] [rbp-B8h]
  unsigned __int64 v44; // [rsp+8h] [rbp-B8h]
  unsigned int v45; // [rsp+10h] [rbp-B0h]
  __int64 v46; // [rsp+10h] [rbp-B0h]
  __int64 v47; // [rsp+18h] [rbp-A8h]
  char v48; // [rsp+18h] [rbp-A8h]
  char v49; // [rsp+18h] [rbp-A8h]
  bool v50; // [rsp+18h] [rbp-A8h]
  __int64 v53; // [rsp+38h] [rbp-88h] BYREF
  const char *v54; // [rsp+40h] [rbp-80h] BYREF
  _BYTE *v55; // [rsp+48h] [rbp-78h]
  _BYTE *v56; // [rsp+50h] [rbp-70h]
  __int64 v57; // [rsp+58h] [rbp-68h]
  int v58; // [rsp+60h] [rbp-60h]
  _BYTE v59[88]; // [rsp+68h] [rbp-58h] BYREF
  unsigned int v60; // [rsp+D0h] [rbp+10h]
  unsigned int v61; // [rsp+D0h] [rbp+10h]

  if ( !a4 )
    return;
  v54 = 0;
  v55 = v59;
  v56 = v59;
  v57 = 4;
  v58 = 0;
  v11 = sub_13F3D10(a1, a3, 1u, (__int64)&v54);
  v12 = a7;
  v13 = v11;
  if ( v56 != v55 )
  {
    v47 = v11;
    _libc_free((unsigned __int64)v56);
    v12 = a7;
    v13 = v47;
  }
  v14 = *(_BYTE *)(v13 + 16);
  if ( v14 == 15 )
  {
    BYTE1(v56) = 1;
    v21 = "Undefined behavior: Null pointer dereference";
    goto LABEL_28;
  }
  if ( v14 == 9 )
  {
    BYTE1(v56) = 1;
    v21 = "Undefined behavior: Undef pointer dereference";
    goto LABEL_28;
  }
  if ( v14 != 13 )
  {
    if ( (v12 & 2) != 0 )
    {
      if ( v14 == 3 )
      {
        if ( (*(_BYTE *)(v13 + 80) & 1) != 0 )
        {
          BYTE1(v56) = 1;
          v21 = "Undefined behavior: Write to read-only memory";
          goto LABEL_28;
        }
        goto LABEL_11;
      }
      if ( (v14 & 0xFB) == 0 )
      {
        BYTE1(v56) = 1;
        v21 = "Undefined behavior: Write to text section";
        goto LABEL_28;
      }
    }
LABEL_42:
    if ( (v12 & 1) != 0 )
    {
      if ( !v14 )
      {
        BYTE1(v56) = 1;
        v21 = "Unusual: Load from function body";
        goto LABEL_28;
      }
      if ( v14 == 4 )
      {
        BYTE1(v56) = 1;
        v21 = "Undefined behavior: Load from block address";
        goto LABEL_28;
      }
      goto LABEL_14;
    }
    goto LABEL_12;
  }
  v24 = *(_DWORD *)(v13 + 32);
  v25 = v13 + 24;
  if ( v24 <= 0x40 )
  {
    v27 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v24) == *(_QWORD *)(v13 + 24);
  }
  else
  {
    v60 = v12;
    v42 = *(_DWORD *)(v13 + 32);
    v26 = sub_16A58F0(v25);
    v24 = v42;
    v12 = v60;
    v25 = v13 + 24;
    v14 = 13;
    v27 = v42 == v26;
  }
  if ( v27 )
  {
    BYTE1(v56) = 1;
    v21 = "Unusual: All-ones pointer dereference";
    goto LABEL_28;
  }
  if ( v24 <= 0x40 )
  {
    v29 = *(_QWORD *)(v13 + 24) == 1;
  }
  else
  {
    v61 = v12;
    v45 = v24;
    v49 = v14;
    v28 = sub_16A57B0(v25);
    v14 = v49;
    v12 = v61;
    v29 = v45 - 1 == v28;
  }
  if ( v29 )
  {
    BYTE1(v56) = 1;
    v21 = "Unusual: Address one pointer dereference";
    goto LABEL_28;
  }
  if ( (v12 & 2) != 0 )
    goto LABEL_42;
LABEL_11:
  if ( (v12 & 1) == 0 )
  {
LABEL_12:
    if ( (v12 & 4) != 0 && v14 == 4 )
    {
      BYTE1(v56) = 1;
      v21 = "Undefined behavior: Call to block address";
      goto LABEL_28;
    }
  }
LABEL_14:
  v48 = (v12 >> 3) & (v14 != 4 && (unsigned __int8)v14 <= 0x10u);
  if ( v48 )
  {
    BYTE1(v56) = 1;
    v21 = "Undefined behavior: Branch to non-blockaddress";
    goto LABEL_28;
  }
  v15 = a1[21];
  v53 = 0;
  v16 = sub_14AC610(a3, &v53, v15);
  v17 = v16;
  if ( !v16 )
    return;
  v18 = *(_BYTE *)(v16 + 16);
  if ( v18 <= 0x17u )
  {
    if ( v18 == 3 && !(unsigned __int8)sub_15E4F60(v17) )
      __asm { jmp     rax }
LABEL_18:
    if ( !a6 )
      return;
    v19 = 0;
    if ( a5 )
      return;
    goto LABEL_20;
  }
  if ( v18 != 53 )
    goto LABEL_18;
  v46 = *(_QWORD *)(v17 + 56);
  v30 = sub_15F8BF0(v17);
  v31 = a4 != -1;
  if ( !v30 )
  {
    v36 = *(unsigned __int8 *)(v46 + 8);
    if ( (unsigned __int8)v36 <= 0xFu )
    {
      v39 = 35454;
      if ( _bittest64(&v39, v36) )
        goto LABEL_83;
    }
    if ( (unsigned int)(v36 - 13) > 1 && (_DWORD)v36 != 16 )
    {
      v19 = (unsigned int)(1 << *(_WORD *)(v17 + 18)) >> 1;
      if ( !v19 )
      {
        v19 = 0;
        goto LABEL_65;
      }
LABEL_89:
      if ( !a6 || a5 )
      {
LABEL_26:
        if ( a5 <= (-(v53 | v19) & (v53 | (unsigned __int64)v19)) )
          return;
        BYTE1(v56) = 1;
        v21 = "Undefined behavior: Memory reference address is misaligned";
LABEL_28:
        v54 = v21;
        LOBYTE(v56) = 3;
        sub_16E2CE0(&v54, a1 + 30);
        v22 = (_BYTE *)a1[33];
        if ( (unsigned __int64)v22 >= a1[32] )
        {
          sub_16E7DE0(a1 + 30, 10);
        }
        else
        {
          a1[33] = v22 + 1;
          *v22 = 10;
        }
        if ( *(_BYTE *)(a2 + 16) <= 0x17u )
        {
          sub_15537D0(a2, a1 + 30, 1);
          v23 = (_BYTE *)a1[33];
          if ( (unsigned __int64)v23 < a1[32] )
            goto LABEL_32;
        }
        else
        {
          sub_155C2B0(a2, a1 + 30, 0);
          v23 = (_BYTE *)a1[33];
          if ( (unsigned __int64)v23 < a1[32] )
          {
LABEL_32:
            a1[33] = v23 + 1;
            *v23 = 10;
            return;
          }
        }
        sub_16E7DE0(a1 + 30, 10);
        return;
      }
      goto LABEL_20;
    }
    v37 = sub_16435F0(v46, 0);
    v31 = a4 != -1;
    if ( v37 )
    {
LABEL_83:
      v50 = v31;
      v38 = sub_12BE0A0(a1[21], v46);
      v31 = v50;
      v32 = v38;
      v48 = v50 && v38 != -1;
      v19 = (unsigned int)(1 << *(_WORD *)(v17 + 18)) >> 1;
      if ( v19 )
        goto LABEL_62;
      goto LABEL_57;
    }
  }
  v19 = (unsigned int)(1 << *(_WORD *)(v17 + 18)) >> 1;
  if ( v19 )
    goto LABEL_89;
  v32 = -1;
LABEL_57:
  v33 = *(unsigned __int8 *)(v46 + 8);
  if ( (unsigned __int8)v33 <= 0xFu )
  {
    v40 = 35454;
    if ( _bittest64(&v40, v33) )
      goto LABEL_88;
  }
  v48 = v31 && v32 != -1;
  if ( (unsigned int)(v33 - 13) <= 1 || (v19 = 0, (_DWORD)v33 == 16) )
  {
    v43 = v32;
    v34 = sub_16435F0(v46, 0);
    v32 = v43;
    if ( v34 )
    {
LABEL_88:
      v44 = v32;
      v41 = sub_15A9FE0(a1[21], v46);
      v32 = v44;
      v19 = v41;
      goto LABEL_62;
    }
    v19 = 0;
  }
LABEL_62:
  if ( v48 && (v53 < 0 || a4 + v53 > v32) )
  {
    BYTE1(v56) = 1;
    v21 = "Undefined behavior: Buffer overflow";
    goto LABEL_28;
  }
LABEL_65:
  if ( a5 || !a6 )
  {
LABEL_25:
    if ( !v19 )
      return;
    goto LABEL_26;
  }
LABEL_20:
  v20 = *(unsigned __int8 *)(a6 + 8);
  if ( (unsigned __int8)v20 <= 0xFu && (v35 = 35454, _bittest64(&v35, v20))
    || ((unsigned int)(v20 - 13) <= 1 || (_DWORD)v20 == 16) && (unsigned __int8)sub_16435F0(a6, 0) )
  {
    a5 = sub_15A9FE0(a1[21], a6);
    goto LABEL_25;
  }
}
