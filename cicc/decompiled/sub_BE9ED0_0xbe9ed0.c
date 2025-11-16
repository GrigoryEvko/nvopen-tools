// Function: sub_BE9ED0
// Address: 0xbe9ed0
//
void __fastcall sub_BE9ED0(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // r14
  __int64 v5; // rdi
  __m128i **v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // r12
  _BYTE *v11; // rax
  const char *v12; // rax
  __int64 v13; // rdx
  const char *v14; // rax
  __int64 v15; // rdx
  const char *v16; // rax
  __int64 v17; // rdx
  const char *v18; // rax
  size_t v19; // rdx
  unsigned __int8 **v20; // rbx
  __int64 v21; // rax
  unsigned __int8 **v22; // r15
  unsigned __int8 *v23; // rax
  __m128i v24; // rax
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  const char *v28; // rax
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // r14
  _BYTE *v33; // rax
  __int64 v34; // rax
  __int64 v35; // rbx
  __int64 *v36; // rdi
  int v37; // esi
  __int64 v38; // r15
  __int64 v39; // rax
  const char *v40; // rax
  const char *v41; // rax
  __int64 v42; // rdx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rdi
  __int64 v46; // rdx
  __m128i **v47; // [rsp+8h] [rbp-138h]
  unsigned __int8 *v48; // [rsp+18h] [rbp-128h] BYREF
  __m128i v49[2]; // [rsp+20h] [rbp-120h] BYREF
  char v50; // [rsp+40h] [rbp-100h]
  char v51; // [rsp+41h] [rbp-FFh]
  __m128i v52; // [rsp+50h] [rbp-F0h] BYREF
  _BYTE v53[16]; // [rsp+60h] [rbp-E0h] BYREF
  __int16 v54; // [rsp+70h] [rbp-D0h]
  __m128i v55; // [rsp+80h] [rbp-C0h] BYREF
  const char *v56; // [rsp+90h] [rbp-B0h]
  __int64 v57; // [rsp+98h] [rbp-A8h]
  __int16 v58; // [rsp+A0h] [rbp-A0h]
  __m128i v59[2]; // [rsp+B0h] [rbp-90h] BYREF
  char v60; // [rsp+D0h] [rbp-70h]
  char v61; // [rsp+D1h] [rbp-6Fh]
  __m128i v62[2]; // [rsp+E0h] [rbp-60h] BYREF
  char v63; // [rsp+100h] [rbp-40h]
  char v64; // [rsp+101h] [rbp-3Fh]

  v3 = a2;
  v4 = *(_QWORD *)(a2 + 24);
  if ( !sub_B2FC80(a2) )
  {
    v5 = *(_QWORD *)(a2 - 32);
    if ( *(_QWORD *)(v5 + 8) != v4 )
    {
      v32 = *(_QWORD *)a1;
      v64 = 1;
      v62[0].m128i_i64[0] = (__int64)"Global variable initializer type does not match global variable type!";
      v63 = 3;
      if ( v32 )
      {
        sub_CA0E80(v62, v32);
        v33 = *(_BYTE **)(v32 + 32);
        if ( (unsigned __int64)v33 >= *(_QWORD *)(v32 + 24) )
        {
          sub_CB5D20(v32, 10);
        }
        else
        {
          *(_QWORD *)(v32 + 32) = v33 + 1;
          *v33 = 10;
        }
        v34 = *(_QWORD *)a1;
        *(_BYTE *)(a1 + 152) = 1;
        if ( v34 )
          sub_BDBD80(a1, (_BYTE *)a2);
      }
      else
      {
        *(_BYTE *)(a1 + 152) = 1;
      }
      return;
    }
    if ( (*(_BYTE *)(a2 + 32) & 0xF) == 0xA )
    {
      if ( !sub_AC30F0(v5) )
      {
        v59[0].m128i_i64[0] = a2;
        v40 = "'common' global must have a zero initializer!";
        v64 = 1;
        goto LABEL_65;
      }
      if ( (*(_BYTE *)(a2 + 80) & 1) != 0 )
      {
        v59[0].m128i_i64[0] = a2;
        v40 = "'common' global may not be marked constant!";
        v64 = 1;
        goto LABEL_65;
      }
      if ( *(_QWORD *)(a2 + 48) )
      {
        v59[0].m128i_i64[0] = a2;
        v40 = "'common' global may not be in a Comdat!";
        v64 = 1;
        goto LABEL_65;
      }
    }
  }
  if ( (*(_BYTE *)(a2 + 7) & 0x10) == 0 )
    goto LABEL_5;
  v12 = sub_BD5D20(a2);
  if ( v13 == 17
    && !(*(_QWORD *)v12 ^ 0x6F6C672E6D766C6CLL | *((_QWORD *)v12 + 1) ^ 0x726F74635F6C6162LL)
    && v12[16] == 115
    || (v14 = sub_BD5D20(a2), v15 == 17)
    && !(*(_QWORD *)v14 ^ 0x6F6C672E6D766C6CLL | *((_QWORD *)v14 + 1) ^ 0x726F74645F6C6162LL)
    && v14[16] == 115 )
  {
    if ( !sub_B2FC80(a2) && (*(_BYTE *)(a2 + 32) & 0xF) != 6 )
      goto LABEL_66;
    if ( *(_QWORD *)(a2 + 16) )
      goto LABEL_64;
    if ( *(_BYTE *)(v4 + 8) == 16 )
    {
      v35 = *(_QWORD *)(v4 + 24);
      v36 = *(__int64 **)(a1 + 144);
      v37 = *(_DWORD *)(*(_QWORD *)(a1 + 136) + 8LL);
      if ( *(_BYTE *)(v35 + 8) != 15 )
      {
        sub_BCE3C0(v36, v37);
        goto LABEL_58;
      }
      v38 = sub_BCE3C0(v36, v37);
      if ( (unsigned int)(*(_DWORD *)(v35 + 12) - 2) > 1 )
        goto LABEL_58;
      a2 = 32;
      if ( !sub_BCAC40(**(_QWORD **)(v35 + 16), 32) )
        goto LABEL_58;
      v39 = *(_QWORD *)(v35 + 16);
      if ( v38 != *(_QWORD *)(v39 + 8) )
        goto LABEL_58;
      if ( *(_DWORD *)(v35 + 12) != 3 )
      {
        v64 = 1;
        v62[0].m128i_i64[0] = (__int64)"the third field of the element type is mandatory, specify ptr null to migrate fro"
                                       "m the obsoleted 2-field form";
        v63 = 3;
        sub_BDBF70((__int64 *)a1, (__int64)v62);
        return;
      }
      if ( *(_BYTE *)(*(_QWORD *)(v39 + 16) + 8LL) != 14 )
        goto LABEL_58;
    }
  }
  if ( (*(_BYTE *)(v3 + 7) & 0x10) == 0
    || ((v16 = sub_BD5D20(v3), v17 != 9) || *(_QWORD *)v16 != 0x6573752E6D766C6CLL || v16[8] != 100)
    && (v18 = sub_BD5D20(v3), a2 = v19, !sub_9691B0(v18, v19, "llvm.compiler.used", 18)) )
  {
LABEL_5:
    v52.m128i_i64[1] = 0x100000000LL;
    v52.m128i_i64[0] = (__int64)v53;
    sub_B91D10(v3, 0, (__int64)&v52);
    v6 = (__m128i **)v52.m128i_i64[0];
    v7 = v52.m128i_i64[0] + 8LL * v52.m128i_u32[2];
    if ( v7 == v52.m128i_i64[0] )
    {
LABEL_8:
      if ( sub_BCEA30(v4) )
      {
        v8 = (__int64)v62;
        v64 = 1;
        v59[0].m128i_i64[0] = v3;
        v62[0].m128i_i64[0] = (__int64)"Globals cannot contain scalable types";
        v63 = 3;
        sub_BE1030((_BYTE *)a1, (__int64)v62, v59);
      }
      else if ( (unsigned __int8)sub_BCEF10(v4) )
      {
        v61 = 1;
        v59[0].m128i_i64[0] = (__int64)" has illegal target extension type";
        v60 = 3;
        v41 = sub_BD5D20(v3);
        v58 = 1283;
        v57 = v42;
        v55.m128i_i64[0] = (__int64)"Global @";
        v56 = v41;
        sub_9C6370(v62, &v55, v59, (__int64)"Global @", v43, v44);
        v8 = (__int64)v62;
        sub_BDBF70((__int64 *)a1, (__int64)v62);
        v45 = *(_QWORD *)a1;
        if ( v4 )
        {
          if ( v45 )
          {
            v8 = v4;
            sub_BD9860(v45, v4);
          }
        }
      }
      else
      {
        if ( !sub_B2FC80(v3) )
        {
          sub_BDC820(a1, *(_QWORD *)(v3 - 32));
          v8 = v3;
          sub_BE9180(a1, v3);
          v9 = v52.m128i_i64[0];
          if ( (_BYTE *)v52.m128i_i64[0] == v53 )
            return;
LABEL_18:
          _libc_free(v9, v8);
          return;
        }
        v8 = v3;
        sub_BE9180(a1, v3);
      }
    }
    else
    {
      while ( 1 )
      {
        v8 = (__int64)*v6;
        v47 = (__m128i **)v7;
        if ( (*v6)->m128i_i8[0] != 8 )
          break;
        ++v6;
        sub_BDDFD0(a1, (const char *)v8);
        v7 = (__int64)v47;
        if ( v47 == v6 )
          goto LABEL_8;
      }
      v10 = *(_QWORD *)a1;
      v64 = 1;
      v62[0].m128i_i64[0] = (__int64)"!dbg attachment of global variable must be a DIGlobalVariableExpression";
      v63 = 3;
      if ( v10 )
      {
        v8 = v10;
        sub_CA0E80(v62, v10);
        v11 = *(_BYTE **)(v10 + 32);
        if ( (unsigned __int64)v11 >= *(_QWORD *)(v10 + 24) )
        {
          v8 = 10;
          sub_CB5D20(v10, 10);
        }
        else
        {
          *(_QWORD *)(v10 + 32) = v11 + 1;
          *v11 = 10;
        }
      }
      *(_BYTE *)(a1 + 152) |= *(_BYTE *)(a1 + 154);
      *(_BYTE *)(a1 + 153) = 1;
    }
    v9 = v52.m128i_i64[0];
    if ( (_BYTE *)v52.m128i_i64[0] == v53 )
      return;
    goto LABEL_18;
  }
  if ( !sub_B2FC80(v3) && (*(_BYTE *)(v3 + 32) & 0xF) != 6 )
  {
LABEL_66:
    v59[0].m128i_i64[0] = v3;
    v40 = "invalid linkage for intrinsic global variable";
    v64 = 1;
    goto LABEL_65;
  }
  if ( *(_QWORD *)(v3 + 16) )
  {
LABEL_64:
    v59[0].m128i_i64[0] = v3;
    v40 = "invalid uses of intrinsic global variable";
    v64 = 1;
    goto LABEL_65;
  }
  if ( *(_BYTE *)(v4 + 8) != 16 )
    goto LABEL_5;
  if ( *(_BYTE *)(*(_QWORD *)(v4 + 24) + 8LL) != 14 )
  {
LABEL_58:
    v59[0].m128i_i64[0] = v3;
    v40 = "wrong type for intrinsic global variable";
    v64 = 1;
LABEL_65:
    v62[0].m128i_i64[0] = (__int64)v40;
    v63 = 3;
    sub_BE1030((_BYTE *)a1, (__int64)v62, v59);
    return;
  }
  if ( sub_B2FC80(v3) )
    goto LABEL_5;
  v20 = *(unsigned __int8 ***)(v3 - 32);
  if ( *(_BYTE *)v20 == 9 )
  {
    v21 = 32LL * (*((_DWORD *)v20 + 1) & 0x7FFFFFF);
    if ( (*((_BYTE *)v20 + 7) & 0x40) != 0 )
    {
      v22 = (unsigned __int8 **)*(v20 - 1);
      v20 = &v22[(unsigned __int64)v21 / 8];
    }
    else
    {
      v22 = &v20[v21 / 0xFFFFFFFFFFFFFFF8LL];
    }
    if ( v22 == v20 )
      goto LABEL_5;
    while ( 1 )
    {
      v23 = sub_BD3990(*v22, a2);
      v48 = v23;
      if ( *v23 != 3 && *v23 > 1u )
        break;
      if ( (v23[7] & 0x10) == 0 )
      {
        v61 = 1;
        v59[0].m128i_i64[0] = (__int64)" must be named";
        v60 = 3;
        v24.m128i_i64[0] = (__int64)sub_BD5D20(v3);
        v51 = 1;
        v54 = 261;
        v52 = v24;
        v28 = "members of ";
        goto LABEL_42;
      }
      v22 += 4;
      if ( v20 == v22 )
        goto LABEL_5;
    }
    v61 = 1;
    v59[0].m128i_i64[0] = (__int64)" member";
    v60 = 3;
    v51 = 1;
    v52.m128i_i64[0] = (__int64)sub_BD5D20(v3);
    v28 = "invalid ";
    v54 = 261;
    v52.m128i_i64[1] = v46;
LABEL_42:
    v49[0].m128i_i64[0] = (__int64)v28;
    v50 = 3;
    sub_9C6370(&v55, v49, &v52, v25, v26, v27);
    sub_9C6370(v62, &v55, v59, v29, v30, v31);
    sub_BE1130((_BYTE *)a1, (__int64)v62, &v48);
  }
  else
  {
    v64 = 1;
    v62[0].m128i_i64[0] = (__int64)"wrong initalizer for intrinsic global variable";
    v63 = 3;
    sub_BDBF70((__int64 *)a1, (__int64)v62);
    if ( *(_QWORD *)a1 )
      sub_BDBD80(a1, v20);
  }
}
