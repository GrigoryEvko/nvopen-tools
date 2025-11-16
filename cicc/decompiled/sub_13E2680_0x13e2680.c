// Function: sub_13E2680
// Address: 0x13e2680
//
__int64 **__fastcall sub_13E2680(_QWORD *a1, unsigned __int8 *a2, unsigned __int8 *a3, __int64 *a4)
{
  int v7; // edi
  int v8; // eax
  __int64 v10; // rdx
  unsigned __int8 *v11; // rsi
  unsigned __int8 *v12; // rdx
  unsigned __int8 v13; // al
  __int64 v14; // rsi
  __int64 v15; // rdx
  int v16; // edx
  unsigned int v17; // edx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 v22; // r9
  unsigned __int8 *v23; // r8
  _QWORD *v24; // r9
  __int64 **result; // rax
  int v26; // eax
  _BYTE *v27; // rdi
  unsigned int v28; // eax
  __int64 v29; // rdx
  size_t v30; // rdx
  __int64 v31; // rcx
  unsigned __int8 **v32; // rax
  unsigned __int8 **v33; // rdx
  unsigned __int8 *v34; // rcx
  unsigned __int64 v35; // rdi
  int v36; // edi
  __int64 *v37; // rcx
  __int64 *v38; // rcx
  __int64 v39; // rdi
  __int64 v40; // rcx
  __int64 v41; // [rsp+0h] [rbp-A0h]
  __int64 v42; // [rsp+8h] [rbp-98h]
  __int64 v43; // [rsp+8h] [rbp-98h]
  __int64 v44; // [rsp+10h] [rbp-90h]
  unsigned int v45; // [rsp+10h] [rbp-90h]
  unsigned __int8 *v46; // [rsp+10h] [rbp-90h]
  __int64 **v47; // [rsp+18h] [rbp-88h]
  __int64 v48; // [rsp+18h] [rbp-88h]
  _BYTE *v49; // [rsp+20h] [rbp-80h] BYREF
  __int64 v50; // [rsp+28h] [rbp-78h]
  _BYTE s[112]; // [rsp+30h] [rbp-70h] BYREF

  if ( a1 == (_QWORD *)a2 )
    return (__int64 **)a3;
  if ( a2[16] <= 0x10u )
    return 0;
  v7 = *((unsigned __int8 *)a1 + 16);
  if ( (unsigned __int8)v7 <= 0x17u )
    return 0;
  v8 = (unsigned __int8)v7;
  if ( (unsigned int)(unsigned __int8)v7 - 35 > 0x11 )
    goto LABEL_18;
  if ( (unsigned __int8)v7 <= 0x2Fu )
  {
    v10 = 0x80A800000000LL;
    if ( _bittest64(&v10, (unsigned __int8)v7) )
    {
      if ( !(unsigned __int8)sub_15F2380(a1) && !(unsigned __int8)sub_15F2370(a1) )
      {
        v8 = *((unsigned __int8 *)a1 + 16);
        v7 = v8;
        goto LABEL_10;
      }
      return 0;
    }
  }
LABEL_10:
  if ( (unsigned __int8)(v7 - 48) <= 1u || (unsigned int)(v8 - 41) <= 1 )
  {
    if ( (unsigned __int8)sub_15F23D0(a1) )
      return 0;
    v7 = *((unsigned __int8 *)a1 + 16);
  }
  v11 = (unsigned __int8 *)*(a1 - 6);
  v12 = (unsigned __int8 *)*(a1 - 3);
  if ( v11 && v11 == a2 )
  {
    v36 = v7 - 24;
    v37 = a4;
    v11 = a3;
    return (__int64 **)sub_13DDBD0(v36, v11, v12, v37, 2u);
  }
  if ( v12 && a2 == v12 )
  {
    v36 = v7 - 24;
    v37 = a4;
    v12 = a3;
    return (__int64 **)sub_13DDBD0(v36, v11, v12, v37, 2u);
  }
LABEL_18:
  v13 = v7 - 75;
  if ( (unsigned __int8)(v7 - 75) <= 1u )
  {
    v14 = *(a1 - 6);
    v15 = *(a1 - 3);
    if ( v14 && (unsigned __int8 *)v14 == a2 )
    {
      v38 = a4;
      v14 = (__int64)a3;
      v39 = *((_WORD *)a1 + 9) & 0x7FFF;
    }
    else
    {
      if ( a2 != (unsigned __int8 *)v15 || !v15 )
        goto LABEL_23;
      v38 = a4;
      v15 = (__int64)a3;
      v39 = *((_WORD *)a1 + 9) & 0x7FFF;
    }
    return sub_13DB900(v39, v14, v15, v38, 2u);
  }
  if ( (_BYTE)v7 == 56 )
  {
    v26 = *((_DWORD *)a1 + 5);
    v49 = s;
    v27 = s;
    v28 = v26 & 0xFFFFFFF;
    v50 = 0x800000000LL;
    v29 = v28;
    if ( v28 > 8uLL )
    {
      v45 = v28;
      v48 = v28;
      sub_16CD150(&v49, s, v28, 8);
      v27 = v49;
      v28 = v45;
      v29 = v48;
    }
    v30 = 8 * v29;
    LODWORD(v50) = v28;
    if ( v30 )
    {
      memset(v27, 0, v30);
      v27 = v49;
    }
    v31 = 24LL * (*((_DWORD *)a1 + 5) & 0xFFFFFFF);
    if ( (*((_BYTE *)a1 + 23) & 0x40) != 0 )
    {
      v32 = (unsigned __int8 **)*(a1 - 1);
      v33 = &v32[(unsigned __int64)v31 / 8];
    }
    else
    {
      v33 = (unsigned __int8 **)a1;
      v32 = (unsigned __int8 **)&a1[v31 / 0xFFFFFFFFFFFFFFF8LL];
    }
    if ( v33 != v32 )
    {
      do
      {
        v34 = *v32;
        if ( a2 == *v32 )
          v34 = a3;
        v32 += 3;
        v27 += 8;
        *((_QWORD *)v27 - 1) = v34;
      }
      while ( v33 != v32 );
      v27 = v49;
    }
    result = (__int64 **)sub_13E1350(a1[7], (__int64 **)v27, (unsigned int)v50, a4);
    v35 = (unsigned __int64)v49;
    if ( v49 == s )
      return result;
    goto LABEL_55;
  }
LABEL_23:
  if ( a3[16] > 0x10u )
    return 0;
  v16 = *((_DWORD *)a1 + 5);
  v49 = s;
  v50 = 0x800000000LL;
  v17 = v16 & 0xFFFFFFF;
  if ( !v17 )
  {
    v24 = s;
    v19 = 0;
    goto LABEL_63;
  }
  v18 = v17;
  v19 = 0;
  v20 = 3 * v18;
  v21 = 0;
  v22 = 8 * v20;
  do
  {
    if ( (*((_BYTE *)a1 + 23) & 0x40) != 0 )
    {
      v23 = *(unsigned __int8 **)(*(a1 - 1) + v21);
      if ( v23 == a2 )
        goto LABEL_27;
    }
    else
    {
      v23 = *(unsigned __int8 **)((char *)a1 + v21 + -24 * (*((_DWORD *)a1 + 5) & 0xFFFFFFF));
      if ( v23 == a2 )
      {
LABEL_27:
        if ( (unsigned int)v19 >= HIDWORD(v50) )
        {
          v42 = v22;
          v44 = v21;
          sub_16CD150(&v49, s, 0, 8);
          v19 = (unsigned int)v50;
          v22 = v42;
          v21 = v44;
        }
        *(_QWORD *)&v49[8 * v19] = a3;
        v19 = (unsigned int)(v50 + 1);
        LODWORD(v50) = v50 + 1;
        goto LABEL_30;
      }
    }
    if ( v23[16] > 0x10u )
      break;
    if ( (unsigned int)v19 >= HIDWORD(v50) )
    {
      v41 = v22;
      v43 = v21;
      v46 = v23;
      sub_16CD150(&v49, s, 0, 8);
      v19 = (unsigned int)v50;
      v22 = v41;
      v21 = v43;
      v23 = v46;
    }
    *(_QWORD *)&v49[8 * v19] = v23;
    v19 = (unsigned int)(v50 + 1);
    LODWORD(v50) = v50 + 1;
LABEL_30:
    v21 += 24;
  }
  while ( v21 != v22 );
  v24 = v49;
  if ( v19 != (*((_DWORD *)a1 + 5) & 0xFFFFFFF) )
  {
    if ( v49 != s )
      _libc_free((unsigned __int64)v49);
    return 0;
  }
  LOBYTE(v7) = *((_BYTE *)a1 + 16);
  v13 = v7 - 75;
LABEL_63:
  v40 = *a4;
  if ( v13 > 1u )
  {
    if ( (_BYTE)v7 == 54 && (*((_BYTE *)a1 + 18) & 1) == 0 )
      result = (__int64 **)sub_14D8290(*v24, *a1, *a4);
    else
      result = (__int64 **)sub_14DD1F0(a1, v24, v19, v40, a4[1]);
  }
  else
  {
    result = (__int64 **)sub_14D7760(*((_WORD *)a1 + 9) & 0x7FFF, *v24, v24[1], v40, a4[1]);
  }
  v35 = (unsigned __int64)v49;
  if ( v49 != s )
  {
LABEL_55:
    v47 = result;
    _libc_free(v35);
    return v47;
  }
  return result;
}
