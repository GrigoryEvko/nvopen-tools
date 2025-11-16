// Function: sub_3763C40
// Address: 0x3763c40
//
unsigned __int8 *__fastcall sub_3763C40(__int64 *a1, __int64 a2, __m128i a3)
{
  unsigned __int16 *v3; // rax
  int v4; // ebx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // rdx
  __int64 v15; // r14
  __int64 v16; // rdx
  unsigned __int16 v17; // bx
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int128 v21; // rax
  __int64 v22; // r9
  __int128 v23; // rax
  __int64 v24; // r9
  unsigned __int8 *result; // rax
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int128 v31; // [rsp+10h] [rbp-A0h]
  unsigned __int8 *v32; // [rsp+10h] [rbp-A0h]
  unsigned int v33; // [rsp+20h] [rbp-90h] BYREF
  __int64 v34; // [rsp+28h] [rbp-88h]
  __int64 v35; // [rsp+30h] [rbp-80h] BYREF
  int v36; // [rsp+38h] [rbp-78h]
  unsigned __int16 v37; // [rsp+40h] [rbp-70h] BYREF
  __int64 v38; // [rsp+48h] [rbp-68h]
  unsigned __int16 v39; // [rsp+50h] [rbp-60h] BYREF
  __int64 v40; // [rsp+58h] [rbp-58h]
  __int64 v41; // [rsp+60h] [rbp-50h]
  __int64 v42; // [rsp+68h] [rbp-48h]
  __int64 v43; // [rsp+70h] [rbp-40h] BYREF
  __int64 v44; // [rsp+78h] [rbp-38h]

  v3 = *(unsigned __int16 **)(a2 + 48);
  v4 = *v3;
  v5 = *((_QWORD *)v3 + 1);
  LOWORD(v33) = v4;
  v34 = v5;
  if ( !(_WORD)v4 )
    return 0;
  v6 = a1[1] + 500LL * (unsigned __int16)v4;
  if ( *(_BYTE *)(v6 + 6605) == 2 || *(_BYTE *)(v6 + 6604) == 2 )
    return 0;
  v8 = *(_QWORD *)(a2 + 80);
  v35 = v8;
  if ( !v8 )
  {
    v36 = *(_DWORD *)(a2 + 72);
    v26 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL);
    v27 = *(_QWORD *)(v26 + 104);
    LOWORD(v26) = *(_WORD *)(v26 + 96);
    v38 = v27;
    v37 = v26;
    goto LABEL_20;
  }
  sub_B96E90((__int64)&v35, v8, 1);
  v4 = (unsigned __int16)v33;
  v36 = *(_DWORD *)(a2 + 72);
  v9 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL);
  v10 = *(_QWORD *)(v9 + 104);
  LOWORD(v9) = *(_WORD *)(v9 + 96);
  v38 = v10;
  v37 = v9;
  if ( (_WORD)v33 )
  {
LABEL_20:
    if ( (unsigned __int16)(v4 - 17) > 0xD3u )
      goto LABEL_7;
    v14 = 0;
    LOWORD(v4) = word_4456580[v4 - 1];
    goto LABEL_8;
  }
  if ( !sub_30070B0((__int64)&v33) )
  {
LABEL_7:
    v14 = v34;
    goto LABEL_8;
  }
  LOWORD(v4) = sub_3009970((__int64)&v33, v8, v11, v12, v13);
LABEL_8:
  LOWORD(v43) = v4;
  v44 = v14;
  if ( (_WORD)v4 )
  {
    if ( (_WORD)v4 == 1 || (unsigned __int16)(v4 - 504) <= 7u )
      goto LABEL_32;
    v15 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v4 - 16];
  }
  else
  {
    v41 = sub_3007260((__int64)&v43);
    LODWORD(v15) = v41;
    v42 = v16;
  }
  v17 = v37;
  if ( v37 )
  {
    if ( (unsigned __int16)(v37 - 17) > 0xD3u )
    {
LABEL_12:
      v18 = v38;
      goto LABEL_13;
    }
    v18 = 0;
    v17 = word_4456580[v37 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v37) )
      goto LABEL_12;
    v17 = sub_3009970((__int64)&v37, v8, v28, v29, v30);
  }
LABEL_13:
  v39 = v17;
  v40 = v18;
  if ( !v17 )
  {
    v19 = sub_3007260((__int64)&v39);
    v43 = v19;
    v44 = v20;
    goto LABEL_15;
  }
  if ( v17 == 1 || (unsigned __int16)(v17 - 504) <= 7u )
LABEL_32:
    BUG();
  v19 = *(_QWORD *)&byte_444C4A0[16 * v17 - 16];
LABEL_15:
  *(_QWORD *)&v21 = sub_3400BD0(*a1, (unsigned int)(v15 - v19), (__int64)&v35, v33, v34, 0, a3, 0);
  v31 = v21;
  *(_QWORD *)&v23 = sub_3406EB0(
                      (_QWORD *)*a1,
                      0xBEu,
                      (__int64)&v35,
                      v33,
                      v34,
                      v22,
                      *(_OWORD *)*(_QWORD *)(a2 + 40),
                      v21);
  result = sub_3406EB0((_QWORD *)*a1, 0xBFu, (__int64)&v35, v33, v34, v24, v23, v31);
  if ( v35 )
  {
    v32 = result;
    sub_B91220((__int64)&v35, v35);
    return v32;
  }
  return result;
}
