// Function: sub_30F09A0
// Address: 0x30f09a0
//
void __fastcall sub_30F09A0(__int64 a1, unsigned __int8 *a2, __int64 *a3, __int16 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int8 v14; // si
  const char *v15; // rax
  _BYTE *v16; // rax
  _BYTE *v17; // rax
  __int64 v18; // rax
  unsigned int v19; // eax
  unsigned int v20; // r8d
  __int64 v21; // rdi
  int v22; // eax
  bool v23; // al
  int v24; // eax
  bool v25; // al
  unsigned int v26; // eax
  __int64 v27; // r10
  unsigned __int8 *v28; // r15
  unsigned __int8 v29; // al
  unsigned __int64 v30; // rdx
  __int16 v31; // cx
  char v32; // r15
  unsigned __int64 v33; // rax
  __int64 v34; // rax
  int v35; // edx
  char v36; // al
  unsigned __int64 v37; // rax
  __int64 v38; // r8
  int v39; // ecx
  __int64 v40; // rax
  unsigned __int64 v41; // rdx
  __int64 v42; // rax
  __int16 v43; // ax
  unsigned __int8 v44; // al
  char v45; // al
  char v46; // al
  int v47; // edx
  unsigned __int64 v48; // rdx
  char v49; // al
  __int64 v50; // [rsp+0h] [rbp-A0h]
  unsigned int v51; // [rsp+0h] [rbp-A0h]
  __int64 v52; // [rsp+0h] [rbp-A0h]
  __int64 v53; // [rsp+0h] [rbp-A0h]
  __int64 v54; // [rsp+8h] [rbp-98h]
  char v55; // [rsp+8h] [rbp-98h]
  char v56; // [rsp+8h] [rbp-98h]
  unsigned __int64 v57; // [rsp+8h] [rbp-98h]
  __int64 v58; // [rsp+10h] [rbp-90h]
  __int64 v59; // [rsp+10h] [rbp-90h]
  __int64 v60; // [rsp+10h] [rbp-90h]
  __int64 v61; // [rsp+10h] [rbp-90h]
  unsigned __int64 v62; // [rsp+10h] [rbp-90h]
  __int64 v63; // [rsp+10h] [rbp-90h]
  __int64 v64; // [rsp+10h] [rbp-90h]
  unsigned int v65; // [rsp+18h] [rbp-88h]
  char v66; // [rsp+1Ch] [rbp-84h]
  char v69; // [rsp+28h] [rbp-78h]
  __int64 v70; // [rsp+28h] [rbp-78h]
  __int64 v71; // [rsp+28h] [rbp-78h]
  unsigned __int64 v72; // [rsp+30h] [rbp-70h] BYREF
  char *v73; // [rsp+38h] [rbp-68h]
  __int64 v74; // [rsp+40h] [rbp-60h]
  int v75; // [rsp+48h] [rbp-58h]
  char v76; // [rsp+4Ch] [rbp-54h]
  char v77; // [rsp+50h] [rbp-50h] BYREF
  char v78; // [rsp+51h] [rbp-4Fh]

  v9 = a3[1];
  v69 = a6;
  if ( v9 != -1 && v9 != 0xBFFFFFFFFFFFFFFELL && (v9 & 0x3FFFFFFFFFFFFFFFLL) == 0 )
    return;
  v10 = *a3;
  v11 = *a3;
  v72 = 0;
  v73 = &v77;
  v74 = 4;
  v75 = 0;
  v76 = 1;
  v12 = sub_30EFD90((__int64 *)a1, v11, 1, (__int64)&v72, a5, a6);
  v13 = v12;
  if ( !v76 )
  {
    v58 = v12;
    _libc_free((unsigned __int64)v73);
    v13 = v58;
  }
  v14 = *(_BYTE *)v13;
  if ( *(_BYTE *)v13 == 20 )
  {
    v78 = 1;
    v15 = "Undefined behavior: Null pointer dereference";
    goto LABEL_10;
  }
  if ( (unsigned int)v14 - 12 <= 1 )
  {
    v78 = 1;
    v15 = "Undefined behavior: Undef pointer dereference";
    goto LABEL_10;
  }
  if ( v14 == 17 )
  {
    v20 = *(_DWORD *)(v13 + 32);
    if ( !v20 )
      goto LABEL_86;
    v21 = v13 + 24;
    if ( v20 <= 0x40 )
    {
      v23 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v20) == *(_QWORD *)(v13 + 24);
    }
    else
    {
      v65 = *(_DWORD *)(v13 + 32);
      v50 = v13;
      v59 = v13 + 24;
      v22 = sub_C445E0(v21);
      v20 = v65;
      v21 = v59;
      v13 = v50;
      v23 = v65 == v22;
    }
    if ( v23 )
    {
LABEL_86:
      v78 = 1;
      v15 = "Unusual: All-ones pointer dereference";
      goto LABEL_10;
    }
    if ( v20 <= 0x40 )
    {
      v25 = *(_QWORD *)(v13 + 24) == 1;
    }
    else
    {
      v51 = v20;
      v54 = v13;
      v24 = sub_C444A0(v21);
      v13 = v54;
      v25 = v51 - 1 == v24;
    }
    if ( v25 )
    {
      v78 = 1;
      v15 = "Unusual: Address one pointer dereference";
      goto LABEL_10;
    }
  }
  if ( (v69 & 2) != 0 )
  {
    if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 8) + 32LL) - 26) <= 1 )
    {
      v18 = *(_QWORD *)(v13 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v18 + 8) - 17 <= 1 )
        v18 = **(_QWORD **)(v18 + 16);
      v19 = *(_DWORD *)(v18 + 8);
      if ( v19 <= 0x17FF && ((1LL << SBYTE1(v19)) & 0xFFFF50) != 0 )
      {
        v78 = 1;
        v15 = "Undefined behavior: Write to memory in const addrspace";
        goto LABEL_10;
      }
    }
    if ( v14 == 3 )
    {
      if ( (*(_BYTE *)(v13 + 80) & 1) != 0 )
      {
        v78 = 1;
        v15 = "Undefined behavior: Write to read-only memory";
        goto LABEL_10;
      }
      goto LABEL_40;
    }
    if ( (v14 & 0xFB) == 0 )
    {
      v78 = 1;
      v15 = "Undefined behavior: Write to text section";
      goto LABEL_10;
    }
  }
  if ( (v69 & 1) != 0 )
  {
    if ( !v14 )
    {
      v78 = 1;
      v15 = "Unusual: Load from function body";
      goto LABEL_10;
    }
    if ( v14 == 4 )
    {
      v78 = 1;
      v15 = "Undefined behavior: Load from block address";
      goto LABEL_10;
    }
  }
  else if ( (v69 & 4) != 0 && v14 == 4 )
  {
    v78 = 1;
    v15 = "Undefined behavior: Call to block address";
    goto LABEL_10;
  }
LABEL_40:
  if ( v14 <= 0x15u && v14 != 4 && (v69 & 8) != 0 )
  {
    v78 = 1;
    v15 = "Undefined behavior: Branch to non-blockaddress";
    goto LABEL_10;
  }
  v70 = *(_QWORD *)(a1 + 16);
  v26 = sub_AE43F0(v70, *(_QWORD *)(v10 + 8));
  v27 = v70;
  LODWORD(v73) = v26;
  if ( v26 > 0x40 )
  {
    sub_C43690((__int64)&v72, 0, 0);
    v27 = v70;
  }
  else
  {
    v72 = 0;
  }
  v28 = sub_BD45C0((unsigned __int8 *)v10, v27, (__int64)&v72, 1, 0, 0, 0, 0);
  if ( (unsigned int)v73 > 0x40 )
  {
    v71 = *(_QWORD *)v72;
    j_j___libc_free_0_0(v72);
  }
  else
  {
    v71 = 0;
    if ( (_DWORD)v73 )
      v71 = (__int64)(v72 << (64 - (unsigned __int8)v73)) >> (64 - (unsigned __int8)v73);
  }
  if ( !v28 )
    return;
  v29 = *v28;
  if ( *v28 <= 0x1Cu )
  {
    if ( v29 != 3 || sub_B2FC80((__int64)v28) || (unsigned __int8)sub_B2F6B0((__int64)v28) || (v28[80] & 2) != 0 )
      goto LABEL_71;
    v38 = *((_QWORD *)v28 + 3);
    v39 = *(unsigned __int8 *)(v38 + 8);
    if ( (_BYTE)v39 == 12 || (unsigned __int8)v39 <= 3u || (_BYTE)v39 == 5 || (v39 & 0xFD) == 4 || (v39 & 0xFB) == 0xA )
      goto LABEL_100;
    if ( (unsigned __int8)(*(_BYTE *)(v38 + 8) - 15) > 3u && v39 != 20 )
    {
      v30 = -1;
      v43 = (*((_WORD *)v28 + 17) >> 1) & 0x3F;
      if ( !v43 )
        goto LABEL_127;
      goto LABEL_114;
    }
    v63 = *((_QWORD *)v28 + 3);
    v46 = sub_BCEBA0(v63, 0);
    v38 = v63;
    if ( v46 )
    {
LABEL_100:
      v61 = v38;
      v52 = *(_QWORD *)(a1 + 16);
      v55 = sub_AE5020(v52, v38);
      v40 = sub_9208B0(v52, v61);
      v73 = (char *)v41;
      v72 = (((unsigned __int64)(v40 + 7) >> 3) + (1LL << v55) - 1) >> v55 << v55;
      v42 = sub_CA1930(&v72);
      v38 = v61;
      v30 = v42;
    }
    else
    {
      v30 = -1;
    }
    v43 = (*((_WORD *)v28 + 17) >> 1) & 0x3F;
    if ( !v43 )
    {
      v39 = *(unsigned __int8 *)(v38 + 8);
      v44 = *(_BYTE *)(v38 + 8);
      if ( (_BYTE)v39 == 12 || v44 <= 3u || v44 == 5 )
      {
LABEL_104:
        v62 = v30;
        v32 = 1;
        v45 = sub_AE5020(*(_QWORD *)(a1 + 16), v38);
        v30 = v62;
        v66 = v45;
        goto LABEL_53;
      }
LABEL_127:
      if ( (v39 & 0xFFFFFFFD) != 4 && (v39 & 0xFFFFFFFB) != 0xA )
      {
        if ( (unsigned int)(v39 - 15) > 3 && v39 != 20
          || (v57 = v30, v64 = v38, v49 = sub_BCEBA0(v38, 0), v38 = v64, v30 = v57, !v49) )
        {
          v32 = 0;
          goto LABEL_53;
        }
      }
      goto LABEL_104;
    }
LABEL_114:
    v32 = 1;
    v66 = v43 - 1;
    goto LABEL_53;
  }
  if ( v29 == 60 )
  {
    v60 = *((_QWORD *)v28 + 9);
    if ( !(unsigned __int8)sub_B4CE70((__int64)v28)
      && ((v47 = *(unsigned __int8 *)(v60 + 8), (_BYTE)v47 == 12)
       || (unsigned __int8)v47 <= 3u
       || (_BYTE)v47 == 5
       || (v47 & 0xFB) == 0xA
       || (v47 & 0xFD) == 4
       || ((unsigned __int8)(*(_BYTE *)(v60 + 8) - 15) <= 3u || v47 == 20) && (unsigned __int8)sub_BCEBA0(v60, 0))
      && !sub_BCEA30(v60) )
    {
      v53 = *(_QWORD *)(a1 + 16);
      v56 = sub_AE5020(v53, v60);
      v72 = sub_9208B0(v53, v60);
      v73 = (char *)v48;
      v30 = ((1LL << v56) + ((v72 + 7) >> 3) - 1) >> v56 << v56;
    }
    else
    {
      v30 = -1;
    }
    v31 = *((_WORD *)v28 + 1);
    v32 = 1;
    _BitScanReverse64(&v33, 1LL << v31);
    v66 = 63 - (v33 ^ 0x3F);
LABEL_53:
    v34 = a3[1];
    if ( v34 != 0xBFFFFFFFFFFFFFFELL && v34 != -1 && (v34 & 0x4000000000000000LL) == 0 && v30 != -1 )
    {
      if ( v71 < 0 || (LOBYTE(v73) = 0, v72 = v34 & 0x3FFFFFFFFFFFFFFFLL, v71 + sub_CA1930(&v72) > v30) )
      {
        v78 = 1;
        v15 = "Undefined behavior: Buffer overflow";
        goto LABEL_10;
      }
    }
    if ( !a5 || HIBYTE(a4) == 1 )
    {
      if ( !v32 || !HIBYTE(a4) )
        return;
      goto LABEL_77;
    }
    goto LABEL_73;
  }
LABEL_71:
  if ( !a5 )
    return;
  v32 = 0;
  v66 = 0;
  if ( HIBYTE(a4) == 1 )
    return;
LABEL_73:
  v35 = *(unsigned __int8 *)(a5 + 8);
  if ( (_BYTE)v35 == 12
    || (unsigned __int8)v35 <= 3u
    || (_BYTE)v35 == 5
    || (v35 & 0xFB) == 0xA
    || (v35 & 0xFD) == 4
    || ((unsigned __int8)(*(_BYTE *)(a5 + 8) - 15) <= 3u || v35 == 20) && (unsigned __int8)sub_BCEBA0(a5, 0) )
  {
    v36 = sub_AE5020(*(_QWORD *)(a1 + 16), a5);
    if ( v32 )
    {
      LOBYTE(a4) = v36;
LABEL_77:
      v37 = (v71 | (1LL << v66)) & -(v71 | (1LL << v66));
      if ( !v37 )
        return;
      _BitScanReverse64(&v37, v37);
      if ( (unsigned __int8)a4 <= (unsigned __int8)(63 - (v37 ^ 0x3F)) )
        return;
      v78 = 1;
      v15 = "Undefined behavior: Memory reference address is misaligned";
LABEL_10:
      v72 = (unsigned __int64)v15;
      v77 = 3;
      sub_CA0E80((__int64)&v72, a1 + 88);
      v16 = *(_BYTE **)(a1 + 120);
      if ( (unsigned __int64)v16 >= *(_QWORD *)(a1 + 112) )
      {
        sub_CB5D20(a1 + 88, 10);
      }
      else
      {
        *(_QWORD *)(a1 + 120) = v16 + 1;
        *v16 = 10;
      }
      if ( *a2 <= 0x1Cu )
      {
        sub_A5BF40(a2, a1 + 88, 1, *(_QWORD *)a1);
        v17 = *(_BYTE **)(a1 + 120);
        if ( (unsigned __int64)v17 < *(_QWORD *)(a1 + 112) )
          goto LABEL_14;
      }
      else
      {
        sub_A69870((__int64)a2, (_BYTE *)(a1 + 88), 0);
        v17 = *(_BYTE **)(a1 + 120);
        if ( (unsigned __int64)v17 < *(_QWORD *)(a1 + 112) )
        {
LABEL_14:
          *(_QWORD *)(a1 + 120) = v17 + 1;
          *v17 = 10;
          return;
        }
      }
      sub_CB5D20(a1 + 88, 10);
    }
  }
}
