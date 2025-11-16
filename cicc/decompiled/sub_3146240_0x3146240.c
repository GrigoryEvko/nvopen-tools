// Function: sub_3146240
// Address: 0x3146240
//
__int64 __fastcall sub_3146240(__int64 a1)
{
  __int64 v1; // rsi
  _QWORD *v2; // r12
  __int64 v3; // rax
  __int64 v4; // rdx
  bool v5; // zf
  __int64 v6; // r9
  __int64 v7; // r14
  __int64 v8; // rdx
  int v9; // eax
  __int64 *v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned __int8 v14; // al
  __int64 v15; // rax
  __int64 v16; // r9
  __int64 v17; // rdx
  int v18; // esi
  _QWORD *v19; // rdi
  __int64 *v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // r12
  __int64 v24; // rax
  __int64 v25; // r9
  __int64 v26; // r12
  __int64 v27; // rdx
  int v28; // eax
  _QWORD *v29; // rdi
  __int64 *v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rax
  _QWORD *v33; // rax
  unsigned int v34; // eax
  __int64 v35; // rdx
  __int64 v36; // rax
  _QWORD *v37; // r14
  __int64 v38; // rax
  __int64 v39; // r9
  __int64 v40; // rdx
  __int64 v41; // r15
  int v42; // eax
  __int64 *v43; // rdx
  __int64 *v44; // rsi
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rax
  unsigned __int64 v50; // rdx
  __int64 v51; // [rsp+8h] [rbp-B8h]
  _QWORD *v52; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v53; // [rsp+18h] [rbp-A8h]
  _BYTE v54[48]; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD *src; // [rsp+50h] [rbp-70h] BYREF
  __int64 v56; // [rsp+58h] [rbp-68h]
  _QWORD v57[12]; // [rsp+60h] [rbp-60h] BYREF

  v1 = 8;
  v2 = (_QWORD *)a1;
  v3 = *(_QWORD *)(a1 + 8);
  v56 = 0x600000000LL;
  v53 = 0x600000000LL;
  src = v57;
  v4 = *(unsigned __int8 *)(v3 + 8);
  LODWORD(v56) = 1;
  v57[0] = v4;
  v5 = *(_BYTE *)(v3 + 8) == 12;
  v52 = v54;
  if ( v5 )
  {
    v34 = *(_DWORD *)(v3 + 8);
    v1 = 16;
    LODWORD(v56) = 2;
    v57[1] = v34 >> 8;
  }
  v7 = sub_CBF760(v57, v1);
  if ( src != v57 )
    _libc_free((unsigned __int64)src);
  v8 = (unsigned int)v53;
  v9 = v53;
  if ( (unsigned int)v53 >= (unsigned __int64)HIDWORD(v53) )
  {
    if ( HIDWORD(v53) < (unsigned __int64)(unsigned int)v53 + 1 )
    {
      v1 = (__int64)v54;
      sub_C8D5F0((__int64)&v52, v54, (unsigned int)v53 + 1LL, 8u, (unsigned int)v53 + 1LL, v6);
      v8 = (unsigned int)v53;
    }
    v52[v8] = v7;
    LODWORD(v53) = v53 + 1;
  }
  else
  {
    v10 = &v52[(unsigned int)v53];
    if ( v10 )
    {
      *v10 = v7;
      v9 = v53;
    }
    LODWORD(v53) = v9 + 1;
  }
  if ( sub_AC30F0(a1) )
  {
    v32 = (unsigned int)v53;
    v18 = v53;
    if ( (unsigned int)v53 >= (unsigned __int64)HIDWORD(v53) )
    {
      if ( HIDWORD(v53) < (unsigned __int64)(unsigned int)v53 + 1 )
      {
        sub_C8D5F0((__int64)&v52, v54, (unsigned int)v53 + 1LL, 8u, (unsigned int)v53 + 1LL, v13);
        v32 = (unsigned int)v53;
      }
      v52[v32] = 78;
      v19 = v52;
      v21 = (unsigned int)(v53 + 1);
      LODWORD(v53) = v53 + 1;
      goto LABEL_15;
    }
    v19 = v52;
    v33 = &v52[(unsigned int)v53];
    if ( v33 )
    {
      *v33 = 78;
      v18 = v53;
      v19 = v52;
    }
    goto LABEL_14;
  }
  v14 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 == 3 )
  {
    v15 = sub_3146000(a1, v1);
    v17 = (unsigned int)v53;
    v18 = v53;
    if ( (unsigned int)v53 >= (unsigned __int64)HIDWORD(v53) )
    {
      if ( HIDWORD(v53) < (unsigned __int64)(unsigned int)v53 + 1 )
      {
        v51 = v15;
        sub_C8D5F0((__int64)&v52, v54, (unsigned int)v53 + 1LL, 8u, (unsigned int)v53 + 1LL, v16);
        v17 = (unsigned int)v53;
        v15 = v51;
      }
      v52[v17] = v15;
      v19 = v52;
      v21 = (unsigned int)(v53 + 1);
      LODWORD(v53) = v53 + 1;
      goto LABEL_15;
    }
    v19 = v52;
    v20 = &v52[(unsigned int)v53];
    if ( v20 )
    {
      *v20 = v15;
      v18 = v53;
      v19 = v52;
    }
LABEL_14:
    v21 = (unsigned int)(v18 + 1);
    LODWORD(v53) = v21;
LABEL_15:
    v22 = sub_CBF760(v19, 8 * v21);
    goto LABEL_16;
  }
  if ( v14 <= 3u )
  {
LABEL_20:
    v24 = sub_3145E30(a1);
    goto LABEL_21;
  }
  v35 = (unsigned int)v14 - 15;
  if ( (unsigned int)v35 <= 1 )
  {
    v1 = 8;
    if ( (unsigned __int8)sub_AC5570(a1, 8u) )
    {
      v49 = sub_AC52D0(a1);
      v24 = sub_3145F20(v49, v50);
LABEL_21:
      v26 = v24;
LABEL_22:
      v27 = (unsigned int)v53;
      v28 = v53;
      if ( (unsigned int)v53 >= (unsigned __int64)HIDWORD(v53) )
      {
        if ( HIDWORD(v53) < (unsigned __int64)(unsigned int)v53 + 1 )
        {
          sub_C8D5F0((__int64)&v52, v54, (unsigned int)v53 + 1LL, 8u, (unsigned int)v53 + 1LL, v25);
          v27 = (unsigned int)v53;
        }
        v52[v27] = v26;
        v29 = v52;
        v31 = (unsigned int)(v53 + 1);
        LODWORD(v53) = v53 + 1;
      }
      else
      {
        v29 = v52;
        v30 = &v52[(unsigned int)v53];
        if ( v30 )
        {
          *v30 = v26;
          v28 = v53;
          v29 = v52;
        }
        v31 = (unsigned int)(v28 + 1);
        LODWORD(v53) = v31;
      }
      v22 = sub_CBF760(v29, 8 * v31);
      goto LABEL_16;
    }
    v14 = *(_BYTE *)a1;
  }
  switch ( v14 )
  {
    case 4u:
      a1 = *(_QWORD *)(a1 - 64);
      goto LABEL_20;
    case 5u:
    case 9u:
    case 0xAu:
    case 0xBu:
      v36 = 4LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
      if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
      {
        v37 = *(_QWORD **)(a1 - 8);
        v2 = &v37[v36];
      }
      else
      {
        v37 = (_QWORD *)(a1 - v36 * 8);
      }
      for ( ; v2 != v37; v37 += 4 )
      {
        v38 = sub_3146240(*v37);
        v40 = (unsigned int)v53;
        v41 = v38;
        v42 = v53;
        if ( (unsigned int)v53 >= (unsigned __int64)HIDWORD(v53) )
        {
          if ( HIDWORD(v53) < (unsigned __int64)(unsigned int)v53 + 1 )
          {
            sub_C8D5F0((__int64)&v52, v54, (unsigned int)v53 + 1LL, 8u, (unsigned int)v53 + 1LL, v39);
            v40 = (unsigned int)v53;
          }
          v52[v40] = v41;
          LODWORD(v53) = v53 + 1;
        }
        else
        {
          v43 = &v52[(unsigned int)v53];
          if ( v43 )
          {
            *v43 = v41;
            v42 = v53;
          }
          LODWORD(v53) = v42 + 1;
        }
      }
      break;
    case 6u:
      a1 = *(_QWORD *)(a1 - 32);
      goto LABEL_20;
    case 0x11u:
      v24 = sub_3145D50((unsigned int *)(a1 + 24), v1, v35, v11, v12, v13);
      goto LABEL_21;
    case 0x12u:
      v44 = (__int64 *)(a1 + 24);
      if ( *(void **)(a1 + 24) == sub_C33340() )
        sub_C3E660((__int64)&src, (__int64)v44);
      else
        sub_C3A850((__int64)&src, v44);
      v26 = sub_3145D50((unsigned int *)&src, (__int64)v44, v45, v46, v47, v48);
      if ( (unsigned int)v56 > 0x40 && src )
        j_j___libc_free_0_0((unsigned __int64)src);
      goto LABEL_22;
    default:
      break;
  }
  v22 = sub_CBF760(v52, 8LL * (unsigned int)v53);
LABEL_16:
  if ( v52 != (_QWORD *)v54 )
    _libc_free((unsigned __int64)v52);
  return v22;
}
