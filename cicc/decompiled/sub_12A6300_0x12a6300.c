// Function: sub_12A6300
// Address: 0x12a6300
//
__int64 __fastcall sub_12A6300(
        __int64 *a1,
        unsigned __int64 a2,
        unsigned int a3,
        char a4,
        unsigned __int64 a5,
        unsigned int a6,
        char a7,
        unsigned __int64 a8)
{
  unsigned __int64 v8; // r10
  __int64 v13; // rdi
  unsigned __int64 v14; // r12
  char v15; // cl
  int v16; // r8d
  __int64 v17; // rax
  _QWORD *v18; // r12
  __int64 v19; // rdi
  unsigned __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rsi
  _QWORD *v23; // rdx
  __int64 v24; // rsi
  char v26; // al
  bool v27; // al
  unsigned __int64 v28; // [rsp+0h] [rbp-80h]
  unsigned int v29; // [rsp+8h] [rbp-78h]
  __int64 v30; // [rsp+10h] [rbp-70h]
  unsigned __int64 *v31; // [rsp+10h] [rbp-70h]
  unsigned __int64 v33; // [rsp+10h] [rbp-70h]
  __int64 v35; // [rsp+28h] [rbp-58h] BYREF
  char v36[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v37; // [rsp+40h] [rbp-40h]

  v8 = a5;
  v13 = a1[4];
  v14 = a8;
  v15 = a7;
  if ( *(_BYTE *)(v13 + 352) )
    goto LABEL_2;
  v26 = sub_12789C0(v13 + 8, a8);
  v8 = a5;
  v15 = a7;
  if ( !v26 )
  {
    v13 = a1[4];
LABEL_2:
    v16 = 1;
    v37 = 257;
    if ( !v15 )
    {
      v16 = unk_4D0463C;
      if ( unk_4D0463C )
      {
        v33 = v8;
        v27 = sub_126A420(v13, v8);
        v13 = a1[4];
        v8 = v33;
        v16 = v27;
      }
    }
    v28 = v8;
    v29 = v16;
    v30 = sub_127A030(v13 + 8, a8, 0);
    v17 = sub_1648A60(64, 1);
    v18 = (_QWORD *)v17;
    if ( v17 )
      sub_15F9210(v17, v30, v28, 0, v29, 0);
    v19 = a1[7];
    if ( v19 )
    {
      v31 = (unsigned __int64 *)a1[8];
      sub_157E9D0(v19 + 40, v18);
      v20 = *v31;
      v21 = v18[3] & 7LL;
      v18[4] = v31;
      v20 &= 0xFFFFFFFFFFFFFFF8LL;
      v18[3] = v20 | v21;
      *(_QWORD *)(v20 + 8) = v18 + 3;
      *v31 = *v31 & 7 | (unsigned __int64)(v18 + 3);
    }
    sub_164B780(v18, v36);
    v22 = a1[6];
    if ( v22 )
    {
      v35 = a1[6];
      sub_1623A60(&v35, v22, 2);
      v23 = v18 + 6;
      if ( v18[6] )
      {
        sub_161E7C0(v18 + 6);
        v23 = v18 + 6;
      }
      v24 = v35;
      v18[6] = v35;
      if ( v24 )
        sub_1623210(&v35, v24, v23);
    }
    sub_15F8F50(v18, a6);
    return sub_12A61B0(a1, (__int64)v18, a2, a3, a4);
  }
  if ( *(_BYTE *)(a8 + 140) == 12 )
  {
    do
      v14 = *(_QWORD *)(v14 + 160);
    while ( *(_BYTE *)(v14 + 140) == 12 );
  }
  return sub_12897E0((__int64)a1, a2, a5, *(_QWORD *)(v14 + 128), a3, a6, a4 | a7);
}
