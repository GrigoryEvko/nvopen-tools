// Function: sub_33FAFB0
// Address: 0x33fafb0
//
unsigned __int8 *__fastcall sub_33FAFB0(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7)
{
  unsigned __int16 v9; // bx
  __int64 v10; // rax
  unsigned __int16 v11; // dx
  __int64 v12; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rdx
  char v16; // si
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // [rsp+0h] [rbp-80h]
  __int64 v23; // [rsp+10h] [rbp-70h] BYREF
  __int64 v24; // [rsp+18h] [rbp-68h]
  unsigned __int16 v25; // [rsp+20h] [rbp-60h] BYREF
  __int64 v26; // [rsp+28h] [rbp-58h]
  __int64 v27; // [rsp+30h] [rbp-50h]
  __int64 v28; // [rsp+38h] [rbp-48h]
  __int64 v29; // [rsp+40h] [rbp-40h]
  __int64 v30; // [rsp+48h] [rbp-38h]

  v9 = a5;
  v10 = *(_QWORD *)(a2 + 48) + 16LL * a3;
  v23 = a5;
  v11 = *(_WORD *)v10;
  v12 = *(_QWORD *)(v10 + 8);
  v24 = a6;
  if ( v11 == (_WORD)a5 )
  {
    if ( (_WORD)a5 || v12 == a6 )
      return sub_33FAF80(a1, 216, a4, (unsigned int)v23, v24, a6, a7);
    v26 = v12;
    v25 = 0;
LABEL_5:
    v29 = sub_3007260((__int64)&v25);
    v14 = v29;
    v30 = v15;
    v16 = v15;
    if ( !v9 )
    {
LABEL_6:
      v22 = v14;
      v17 = sub_3007260((__int64)&v23);
      v14 = v22;
      v18 = v17;
      v20 = v19;
      v27 = v18;
      v21 = v18;
      v28 = v20;
      goto LABEL_7;
    }
    goto LABEL_14;
  }
  v25 = v11;
  v26 = v12;
  if ( !v11 )
    goto LABEL_5;
  if ( v11 == 1 || (unsigned __int16)(v11 - 504) <= 7u )
LABEL_19:
    BUG();
  v14 = *(_QWORD *)&byte_444C4A0[16 * v11 - 16];
  v16 = byte_444C4A0[16 * v11 - 8];
  if ( !(_WORD)a5 )
    goto LABEL_6;
LABEL_14:
  if ( v9 == 1 || (unsigned __int16)(v9 - 504) <= 7u )
    goto LABEL_19;
  v21 = *(_QWORD *)&byte_444C4A0[16 * v9 - 16];
  LOBYTE(v20) = byte_444C4A0[16 * v9 - 8];
LABEL_7:
  if ( !(_BYTE)v20 && v16 || v14 >= v21 )
    return sub_33FAF80(a1, 216, a4, (unsigned int)v23, v24, a6, a7);
  return sub_33FAF80(a1, 215, a4, (unsigned int)v23, v24, a6, a7);
}
