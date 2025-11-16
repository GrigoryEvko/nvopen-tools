// Function: sub_3406EE0
// Address: 0x3406ee0
//
unsigned __int8 *__fastcall sub_3406EE0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7)
{
  unsigned __int16 v11; // bx
  __int64 v12; // rax
  unsigned __int16 v13; // dx
  __int64 v14; // rax
  __int128 v15; // rax
  __int64 v16; // r9
  unsigned __int64 v18; // rcx
  __int64 v19; // rdx
  char v20; // si
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int128 v26; // [rsp-20h] [rbp-A0h]
  unsigned __int64 v27; // [rsp+0h] [rbp-80h]
  __int64 v28; // [rsp+10h] [rbp-70h] BYREF
  __int64 v29; // [rsp+18h] [rbp-68h]
  unsigned __int16 v30; // [rsp+20h] [rbp-60h] BYREF
  __int64 v31; // [rsp+28h] [rbp-58h]
  __int64 v32; // [rsp+30h] [rbp-50h]
  __int64 v33; // [rsp+38h] [rbp-48h]
  __int64 v34; // [rsp+40h] [rbp-40h]
  __int64 v35; // [rsp+48h] [rbp-38h]

  v11 = a5;
  v12 = *(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3;
  v28 = a5;
  v13 = *(_WORD *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  v29 = a6;
  if ( v13 == (_WORD)a5 )
  {
    if ( (_WORD)a5 || v14 == a6 )
    {
LABEL_3:
      *(_QWORD *)&v15 = sub_3400D50((__int64)a1, 0, a4, 1u, a7);
      *((_QWORD *)&v26 + 1) = a3;
      *(_QWORD *)&v26 = a2;
      return sub_3406EB0(a1, 0xE6u, a4, (unsigned int)v28, v29, v16, v26, v15);
    }
    v31 = v14;
    v30 = 0;
LABEL_5:
    v34 = sub_3007260((__int64)&v30);
    v18 = v34;
    v35 = v19;
    v20 = v19;
    if ( !v11 )
    {
LABEL_6:
      v27 = v18;
      v21 = sub_3007260((__int64)&v28);
      v18 = v27;
      v22 = v21;
      v24 = v23;
      v32 = v22;
      v25 = v22;
      v33 = v24;
      goto LABEL_7;
    }
    goto LABEL_14;
  }
  v30 = v13;
  v31 = v14;
  if ( !v13 )
    goto LABEL_5;
  if ( v13 == 1 || (unsigned __int16)(v13 - 504) <= 7u )
LABEL_19:
    BUG();
  v18 = *(_QWORD *)&byte_444C4A0[16 * v13 - 16];
  v20 = byte_444C4A0[16 * v13 - 8];
  if ( !(_WORD)a5 )
    goto LABEL_6;
LABEL_14:
  if ( v11 == 1 || (unsigned __int16)(v11 - 504) <= 7u )
    goto LABEL_19;
  v25 = *(_QWORD *)&byte_444C4A0[16 * v11 - 16];
  LOBYTE(v24) = byte_444C4A0[16 * v11 - 8];
LABEL_7:
  if ( !(_BYTE)v24 && v20 || v18 >= v25 )
    goto LABEL_3;
  return sub_33FAF80((__int64)a1, 233, a4, (unsigned int)v28, v29, a6, a7);
}
