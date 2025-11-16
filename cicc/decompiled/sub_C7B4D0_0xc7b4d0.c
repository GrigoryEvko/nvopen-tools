// Function: sub_C7B4D0
// Address: 0xc7b4d0
//
__int64 __fastcall sub_C7B4D0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v6; // r15d
  int v7; // ebx
  unsigned int v8; // esi
  int v9; // eax
  unsigned int v10; // ebx
  __int64 v11; // r8
  __int64 v12; // rcx
  unsigned int v13; // r10d
  _QWORD *v14; // r9
  int v15; // eax
  unsigned int v16; // r8d
  __int64 v17; // rbx
  __int64 v18; // rcx
  __int64 v19; // rdi
  __int64 v20; // rax
  unsigned int v21; // eax
  unsigned __int64 v22; // rbx
  unsigned int v23; // r8d
  unsigned int v24; // esi
  __int64 v26; // rdi
  __int64 v27; // rax
  unsigned int v28; // eax
  unsigned __int64 v29; // r8
  unsigned int v30; // esi
  unsigned int v31; // edx
  unsigned int v32; // ebx
  __int64 v33; // rax
  __int64 v34; // rdx
  unsigned int v35; // edx
  __int64 v36; // rsi
  __int64 v37; // rax
  __int64 v38; // rdx
  char v39; // cl
  unsigned __int64 v40; // rdi
  unsigned __int64 v41; // rax
  char v42; // cl
  unsigned __int64 v43; // rdi
  unsigned __int64 v44; // rax
  unsigned int v45; // eax
  __int64 v46; // rdx
  __int64 v47; // rax
  unsigned __int64 v48; // rax
  unsigned __int64 v49; // rax
  bool v50; // cc
  unsigned int v51; // eax
  __int64 v52; // rdx
  __int64 v53; // rax
  unsigned __int64 v54; // rax
  unsigned int v55; // eax
  unsigned int v56; // eax
  unsigned __int64 v57; // rax
  unsigned __int64 v58; // rax
  unsigned int v59; // [rsp+Ch] [rbp-84h]
  __int64 v60; // [rsp+10h] [rbp-80h]
  unsigned int v61; // [rsp+10h] [rbp-80h]
  unsigned int v62; // [rsp+18h] [rbp-78h]
  __int64 v63; // [rsp+18h] [rbp-78h]
  unsigned __int64 v64; // [rsp+20h] [rbp-70h]
  __int64 v65; // [rsp+20h] [rbp-70h]
  unsigned int v66; // [rsp+20h] [rbp-70h]
  _QWORD *v67; // [rsp+20h] [rbp-70h]
  unsigned int v68; // [rsp+20h] [rbp-70h]
  const void **v69; // [rsp+28h] [rbp-68h]
  unsigned int v70; // [rsp+28h] [rbp-68h]
  _QWORD *v71; // [rsp+28h] [rbp-68h]
  _QWORD *v72; // [rsp+28h] [rbp-68h]
  unsigned __int64 v73; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v74; // [rsp+38h] [rbp-58h]
  unsigned __int64 v75; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v76; // [rsp+48h] [rbp-48h]
  unsigned __int64 v77; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v78; // [rsp+58h] [rbp-38h]

  sub_C7AF00(a1, a2, a3);
  v6 = *(_DWORD *)(a3 + 8);
  if ( v6 > 0x40 )
    v7 = sub_C44630(a3);
  else
    v7 = sub_39FAC40(*(_QWORD *)a3);
  v8 = *(_DWORD *)(a3 + 24);
  v69 = (const void **)(a3 + 16);
  if ( v8 > 0x40 )
  {
    v9 = sub_C44630((__int64)v69);
    if ( v6 == v9 + v7 && v9 == 1 )
    {
      v78 = v8;
      sub_C43780((__int64)&v77, v69);
LABEL_48:
      sub_C46F20((__int64)&v77, 1u);
      v31 = *(_DWORD *)(a2 + 8);
      v32 = v78;
      v74 = v78;
      v73 = v77;
      v33 = 1LL << ((unsigned __int8)v31 - 1);
      if ( v31 > 0x40 )
        v34 = *(_QWORD *)(*(_QWORD *)a2 + 8LL * ((v31 - 1) >> 6));
      else
        v34 = *(_QWORD *)a2;
      if ( (v34 & v33) != 0 )
      {
        v76 = v78;
        if ( v78 > 0x40 )
        {
LABEL_121:
          sub_C43780((__int64)&v75, (const void **)&v73);
          v32 = v76;
          goto LABEL_103;
        }
      }
      else
      {
        if ( v78 > 0x40 )
        {
          if ( !(unsigned __int8)sub_C446F0((__int64 *)&v73, (__int64 *)a2) )
            goto LABEL_53;
          v76 = v32;
          goto LABEL_121;
        }
        if ( (v77 & ~*(_QWORD *)a2) != 0 )
          goto LABEL_53;
        v76 = v78;
      }
      v75 = v77;
LABEL_103:
      if ( v32 > 0x40 )
      {
        sub_C43D10((__int64)&v75);
        v54 = v75;
        v32 = v76;
      }
      else
      {
        v54 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v32) & ~v75;
        if ( !v32 )
          v54 = 0;
        v75 = v54;
      }
      v50 = *(_DWORD *)(a1 + 8) <= 0x40u;
      v78 = v32;
      v77 = v54;
      v76 = 0;
      if ( v50 )
      {
        *(_QWORD *)a1 |= v54;
      }
      else
      {
        sub_C43BD0((_QWORD *)a1, (__int64 *)&v77);
        v32 = v78;
      }
      if ( v32 > 0x40 && v77 )
        j_j___libc_free_0_0(v77);
      if ( v76 > 0x40 && v75 )
        j_j___libc_free_0_0(v75);
      v32 = v74;
LABEL_53:
      v35 = *(_DWORD *)(a2 + 24);
      v36 = *(_QWORD *)(a2 + 16);
      v37 = 1LL << ((unsigned __int8)v35 - 1);
      if ( v35 > 0x40 )
        v38 = *(_QWORD *)(v36 + 8LL * ((v35 - 1) >> 6));
      else
        v38 = *(_QWORD *)(a2 + 16);
      if ( (v38 & v37) == 0 )
        goto LABEL_89;
      if ( v32 <= 0x40 )
      {
        v58 = v73;
        if ( (v73 & v36) == 0 )
          return a1;
      }
      else
      {
        if ( !(unsigned __int8)sub_C446A0((__int64 *)&v73, (__int64 *)(a2 + 16)) )
        {
LABEL_58:
          if ( v73 )
            j_j___libc_free_0_0(v73);
          return a1;
        }
        v76 = v32;
        sub_C43780((__int64)&v75, (const void **)&v73);
        v32 = v76;
        if ( v76 > 0x40 )
        {
          sub_C43D10((__int64)&v75);
          v32 = v76;
          v49 = v75;
LABEL_80:
          v50 = *(_DWORD *)(a1 + 24) <= 0x40u;
          v78 = v32;
          v77 = v49;
          v76 = 0;
          if ( v50 )
          {
            *(_QWORD *)(a1 + 16) |= v49;
          }
          else
          {
            sub_C43BD0((_QWORD *)(a1 + 16), (__int64 *)&v77);
            v32 = v78;
          }
          if ( v32 > 0x40 && v77 )
            j_j___libc_free_0_0(v77);
          if ( v76 > 0x40 && v75 )
            j_j___libc_free_0_0(v75);
          v32 = v74;
LABEL_89:
          if ( v32 <= 0x40 )
            return a1;
          goto LABEL_58;
        }
        v58 = v75;
      }
      v49 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v32) & ~v58;
      if ( !v32 )
        v49 = 0;
      v75 = v49;
      goto LABEL_80;
    }
  }
  else
  {
    v64 = *(_QWORD *)(a3 + 16);
    if ( v6 == (unsigned int)sub_39FAC40(v64) + v7 && v64 && (v64 & (v64 - 1)) == 0 )
    {
      v78 = v8;
      v77 = v64;
      goto LABEL_48;
    }
  }
  v10 = *(_DWORD *)(a2 + 24);
  v11 = *(_QWORD *)(a2 + 16);
  if ( v10 <= 0x40 )
    v12 = *(_QWORD *)(a2 + 16);
  else
    v12 = *(_QWORD *)(v11 + 8LL * ((v10 - 1) >> 6));
  if ( (v12 & (1LL << ((unsigned __int8)v10 - 1))) == 0 )
    goto LABEL_15;
  v13 = *(_DWORD *)(a1 + 24);
  v14 = (_QWORD *)(a1 + 16);
  if ( v13 <= 0x40 )
  {
    if ( *(_QWORD *)(a1 + 16) )
      goto LABEL_33;
LABEL_15:
    v16 = *(_DWORD *)(a2 + 8);
    v17 = *(_QWORD *)a2;
    if ( v16 > 0x40 )
      v18 = *(_QWORD *)(v17 + 8LL * ((v16 - 1) >> 6));
    else
      v18 = *(_QWORD *)a2;
    if ( (v18 & (1LL << ((unsigned __int8)v16 - 1))) == 0 )
      return a1;
    v19 = *(_QWORD *)a3;
    v20 = 1LL << ((unsigned __int8)v6 - 1);
    if ( v6 <= 0x40 )
    {
      if ( (v20 & v19) != 0 )
      {
        if ( v6 )
        {
          v39 = 64 - v6;
          v6 = 64;
          v40 = ~(v19 << v39);
          if ( v40 )
          {
            _BitScanReverse64(&v41, v40);
            v6 = v41 ^ 0x3F;
          }
        }
        goto LABEL_21;
      }
    }
    else if ( (*(_QWORD *)(v19 + 8LL * ((v6 - 1) >> 6)) & v20) != 0 )
    {
      v70 = *(_DWORD *)(a2 + 8);
      v21 = sub_C44500(a3);
      v16 = v70;
      v6 = v21;
      goto LABEL_21;
    }
    v46 = *(_QWORD *)(a3 + 16);
    v47 = 1LL << ((unsigned __int8)v8 - 1);
    if ( v8 > 0x40 )
    {
      if ( (*(_QWORD *)(v46 + 8LL * ((v8 - 1) >> 6)) & v47) != 0 )
      {
        v68 = *(_DWORD *)(a2 + 8);
        v56 = sub_C44500((__int64)v69);
        v16 = v68;
        v6 = v56;
        goto LABEL_21;
      }
    }
    else if ( (v47 & v46) != 0 )
    {
      if ( v8 )
      {
        v6 = 64;
        if ( v46 << (64 - (unsigned __int8)v8) != -1 )
        {
          _BitScanReverse64(&v48, ~(v46 << (64 - (unsigned __int8)v8)));
          v6 = v48 ^ 0x3F;
        }
      }
      else
      {
        v6 = 0;
      }
      goto LABEL_21;
    }
    v6 = 1;
LABEL_21:
    if ( v16 > 0x40 )
    {
      v45 = sub_C44500(a2);
      if ( v6 < v45 )
        v6 = v45;
    }
    else if ( v16 )
    {
      v22 = ~(v17 << (64 - (unsigned __int8)v16));
      if ( v22 )
      {
        _BitScanReverse64(&v22, v22);
        LODWORD(v22) = v22 ^ 0x3F;
        if ( v6 < (unsigned int)v22 )
          v6 = v22;
      }
      else if ( v6 < 0x40 )
      {
        v6 = 64;
      }
    }
    v23 = *(_DWORD *)(a1 + 8);
    v24 = v23 - v6;
    if ( v23 != v23 - v6 )
    {
      if ( v24 > 0x3F || v23 > 0x40 )
        sub_C43C90((_QWORD *)a1, v24, v23);
      else
        *(_QWORD *)a1 |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v6) << v24;
    }
    return a1;
  }
  v59 = *(_DWORD *)(a1 + 24);
  v60 = *(_QWORD *)(a2 + 16);
  v15 = sub_C444A0(a1 + 16);
  v13 = v59;
  v14 = (_QWORD *)(a1 + 16);
  v11 = v60;
  if ( v59 == v15 )
    goto LABEL_15;
LABEL_33:
  v26 = *(_QWORD *)a3;
  v27 = 1LL << ((unsigned __int8)v6 - 1);
  if ( v6 <= 0x40 )
  {
    if ( (v26 & v27) != 0 )
    {
      if ( v6 )
      {
        v42 = 64 - v6;
        v6 = 64;
        v43 = ~(v26 << v42);
        if ( v43 )
        {
          _BitScanReverse64(&v44, v43);
          v6 = v44 ^ 0x3F;
        }
      }
      goto LABEL_36;
    }
  }
  else if ( (*(_QWORD *)(v26 + 8LL * ((v6 - 1) >> 6)) & v27) != 0 )
  {
    v62 = v13;
    v65 = v11;
    v71 = v14;
    v28 = sub_C44500(a3);
    v14 = v71;
    v11 = v65;
    v13 = v62;
    v6 = v28;
    goto LABEL_36;
  }
  v52 = *(_QWORD *)(a3 + 16);
  v53 = 1LL << ((unsigned __int8)v8 - 1);
  if ( v8 > 0x40 )
  {
    if ( (*(_QWORD *)(v52 + 8LL * ((v8 - 1) >> 6)) & v53) != 0 )
    {
      v61 = v13;
      v63 = v11;
      v67 = v14;
      v55 = sub_C44500((__int64)v69);
      v14 = v67;
      v11 = v63;
      v13 = v61;
      v6 = v55;
      goto LABEL_36;
    }
  }
  else if ( (v53 & v52) != 0 )
  {
    if ( v8 )
    {
      v6 = 64;
      if ( v52 << (64 - (unsigned __int8)v8) != -1 )
      {
        _BitScanReverse64(&v57, ~(v52 << (64 - (unsigned __int8)v8)));
        v6 = v57 ^ 0x3F;
      }
    }
    else
    {
      v6 = 0;
    }
    goto LABEL_36;
  }
  v6 = 1;
LABEL_36:
  if ( v10 > 0x40 )
  {
    v66 = v13;
    v72 = v14;
    v51 = sub_C44500(a2 + 16);
    v13 = v66;
    v14 = v72;
    if ( v6 < v51 )
      v6 = v51;
  }
  else if ( v10 )
  {
    v29 = ~(v11 << (64 - (unsigned __int8)v10));
    if ( v29 )
    {
      _BitScanReverse64(&v29, v29);
      LODWORD(v29) = v29 ^ 0x3F;
      if ( v6 < (unsigned int)v29 )
        v6 = v29;
    }
    else if ( v6 < 0x40 )
    {
      v6 = 64;
    }
  }
  v30 = v13 - v6;
  if ( v13 - v6 != v13 )
  {
    if ( v30 > 0x3F || v13 > 0x40 )
      sub_C43C90(v14, v30, v13);
    else
      *(_QWORD *)(a1 + 16) |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v6) << v30;
  }
  return a1;
}
