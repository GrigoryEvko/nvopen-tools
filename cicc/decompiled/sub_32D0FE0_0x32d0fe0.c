// Function: sub_32D0FE0
// Address: 0x32d0fe0
//
__int64 __fastcall sub_32D0FE0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // r12
  __int64 v5; // rbx
  unsigned __int16 *v6; // rdx
  int v7; // eax
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdx
  unsigned __int64 v12; // rdx
  __int64 v13; // rbx
  unsigned __int16 v14; // ax
  __int64 v15; // rdx
  unsigned int v16; // r12d
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  unsigned int v21; // eax
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v24; // [rsp+18h] [rbp-68h]
  unsigned __int16 v25; // [rsp+20h] [rbp-60h] BYREF
  __int64 v26; // [rsp+28h] [rbp-58h]
  unsigned __int64 v27; // [rsp+30h] [rbp-50h] BYREF
  __int64 v28; // [rsp+38h] [rbp-48h]
  __int64 v29; // [rsp+40h] [rbp-40h]
  __int64 v30; // [rsp+48h] [rbp-38h]

  v4 = a2;
  v5 = 16LL * a3;
  v6 = (unsigned __int16 *)(v5 + *(_QWORD *)(a2 + 48));
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  v25 = v7;
  v26 = v8;
  if ( (_WORD)v7 )
  {
    if ( (unsigned __int16)(v7 - 17) > 0xD3u )
    {
      LOWORD(v27) = v7;
      v28 = v8;
LABEL_4:
      if ( (_WORD)v7 == 1 || (unsigned __int16)(v7 - 504) <= 7u )
        BUG();
      v9 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v7 - 16];
      v24 = v9;
      if ( (unsigned int)v9 > 0x40 )
        goto LABEL_7;
      goto LABEL_11;
    }
    LOWORD(v7) = word_4456580[v7 - 1];
    v10 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v25) )
    {
      v28 = v8;
      LOWORD(v27) = 0;
      goto LABEL_10;
    }
    LOWORD(v7) = sub_3009970((__int64)&v25, a2, v18, v19, v20);
  }
  LOWORD(v27) = v7;
  v28 = v10;
  if ( (_WORD)v7 )
    goto LABEL_4;
LABEL_10:
  v9 = sub_3007260((__int64)&v27);
  v29 = v9;
  v30 = v11;
  v24 = v9;
  if ( (unsigned int)v9 > 0x40 )
  {
LABEL_7:
    a2 = -1;
    sub_C43690((__int64)&v23, -1, 1);
    goto LABEL_14;
  }
LABEL_11:
  v12 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v9;
  if ( !(_DWORD)v9 )
    v12 = 0;
  v23 = v12;
LABEL_14:
  v13 = *(_QWORD *)(v4 + 48) + v5;
  v14 = *(_WORD *)v13;
  v15 = *(_QWORD *)(v13 + 8);
  v25 = v14;
  v26 = v15;
  if ( v14 )
  {
    if ( (unsigned __int16)(v14 - 17) > 0x9Eu )
    {
LABEL_16:
      LODWORD(v28) = 1;
      v27 = 1;
      goto LABEL_17;
    }
LABEL_35:
    v21 = word_4456340[v25 - 1];
    LODWORD(v28) = v21;
    if ( v21 > 0x40 )
      goto LABEL_36;
    goto LABEL_29;
  }
  if ( !sub_30070D0((__int64)&v25) )
    goto LABEL_16;
  if ( sub_3007100((__int64)&v25) )
  {
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( v25 )
    {
      if ( (unsigned __int16)(v25 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      goto LABEL_35;
    }
  }
  v21 = sub_3007130((__int64)&v25, a2);
  LODWORD(v28) = v21;
  if ( v21 > 0x40 )
  {
LABEL_36:
    sub_C43690((__int64)&v27, -1, 1);
    goto LABEL_17;
  }
LABEL_29:
  v22 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v21;
  if ( !v21 )
    v22 = 0;
  v27 = v22;
LABEL_17:
  v16 = sub_32D0760(a1, v4, a3, (int)&v23, (int)&v27, 0);
  if ( (unsigned int)v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  if ( v24 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
  return v16;
}
