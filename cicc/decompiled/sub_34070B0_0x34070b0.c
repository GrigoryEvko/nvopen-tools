// Function: sub_34070B0
// Address: 0x34070b0
//
unsigned __int8 *__fastcall sub_34070B0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7)
{
  __int64 v10; // rax
  __int16 v11; // dx
  __int64 v12; // rcx
  unsigned __int16 v13; // ax
  unsigned __int8 *result; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rax
  int v19; // ebx
  __int64 v20; // r8
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned __int64 v25; // rax
  __int128 v26; // rax
  __int64 v27; // r9
  bool v28; // al
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int16 v31; // ax
  __int64 v32; // rdx
  __int128 v33; // [rsp-30h] [rbp-D0h]
  unsigned int v34; // [rsp+Ch] [rbp-94h]
  int v35; // [rsp+10h] [rbp-90h]
  unsigned __int8 *v36; // [rsp+10h] [rbp-90h]
  int v37; // [rsp+10h] [rbp-90h]
  _QWORD v38[2]; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v39; // [rsp+30h] [rbp-70h] BYREF
  __int64 v40; // [rsp+38h] [rbp-68h]
  unsigned __int64 v41; // [rsp+40h] [rbp-60h] BYREF
  __int64 v42; // [rsp+48h] [rbp-58h]
  __int64 v43; // [rsp+50h] [rbp-50h]
  __int64 v44; // [rsp+58h] [rbp-48h]
  __int64 v45; // [rsp+60h] [rbp-40h] BYREF
  __int64 v46; // [rsp+68h] [rbp-38h]

  v10 = *(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3;
  v38[0] = a5;
  v38[1] = a6;
  v11 = *(_WORD *)v10;
  v12 = *(_QWORD *)(v10 + 8);
  v13 = a5;
  LOWORD(v39) = v11;
  v40 = v12;
  if ( v11 == (_WORD)a5 && ((_WORD)a5 || v12 == a6) )
    return (unsigned __int8 *)a2;
  if ( (_WORD)a5 )
  {
    if ( (unsigned __int16)(a5 - 17) > 0xD3u )
    {
      LOWORD(v45) = a5;
      v46 = a6;
      goto LABEL_24;
    }
    v13 = word_4456580[(unsigned __int16)a5 - 1];
    v32 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)v38) )
    {
      v46 = a6;
      LOWORD(v45) = 0;
      goto LABEL_9;
    }
    v13 = sub_3009970((__int64)v38, a2, v15, v16, v17);
  }
  LOWORD(v45) = v13;
  v46 = v32;
  if ( !v13 )
  {
LABEL_9:
    v18 = sub_3007260((__int64)&v45);
    v19 = (unsigned __int16)v39;
    v43 = v18;
    LODWORD(v20) = v18;
    v44 = v21;
    if ( (_WORD)v39 )
      goto LABEL_10;
LABEL_27:
    v34 = v20;
    v28 = sub_30070B0((__int64)&v39);
    LODWORD(v20) = v34;
    if ( v28 )
    {
      v31 = sub_3009970((__int64)&v39, a2, v29, v30, v34);
      LODWORD(v20) = v34;
      LOWORD(v19) = v31;
LABEL_12:
      LOWORD(v41) = v19;
      v42 = v22;
      if ( !(_WORD)v19 )
        goto LABEL_13;
      goto LABEL_34;
    }
LABEL_11:
    v22 = v40;
    goto LABEL_12;
  }
LABEL_24:
  if ( v13 == 1 || (unsigned __int16)(v13 - 504) <= 7u )
    goto LABEL_40;
  v19 = (unsigned __int16)v39;
  v20 = *(_QWORD *)&byte_444C4A0[16 * v13 - 16];
  if ( !(_WORD)v39 )
    goto LABEL_27;
LABEL_10:
  if ( (unsigned __int16)(v19 - 17) > 0xD3u )
    goto LABEL_11;
  v42 = 0;
  LOWORD(v19) = word_4456580[v19 - 1];
  LOWORD(v41) = v19;
  if ( (_WORD)v19 )
  {
LABEL_34:
    if ( (_WORD)v19 != 1 && (unsigned __int16)(v19 - 504) > 7u )
    {
      LODWORD(v42) = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v19 - 16];
      if ( (unsigned int)v42 <= 0x40 )
        goto LABEL_14;
LABEL_37:
      v37 = v20;
      sub_C43690((__int64)&v41, 0, 0);
      LODWORD(v20) = v37;
      goto LABEL_15;
    }
LABEL_40:
    BUG();
  }
LABEL_13:
  v35 = v20;
  v23 = sub_3007260((__int64)&v41);
  LODWORD(v20) = v35;
  v45 = v23;
  v46 = v24;
  LODWORD(v42) = v23;
  if ( (unsigned int)v23 > 0x40 )
    goto LABEL_37;
LABEL_14:
  v41 = 0;
LABEL_15:
  if ( (_DWORD)v20 )
  {
    if ( (unsigned int)v20 > 0x40 )
    {
      sub_C43C90(&v41, 0, v20);
    }
    else
    {
      v25 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v20);
      if ( (unsigned int)v42 > 0x40 )
        *(_QWORD *)v41 |= v25;
      else
        v41 |= v25;
    }
  }
  *(_QWORD *)&v26 = sub_34007B0((__int64)a1, (__int64)&v41, a4, v39, v40, 0, a7, 0);
  *((_QWORD *)&v33 + 1) = a3;
  *(_QWORD *)&v33 = a2;
  result = sub_3406EB0(a1, 0xBAu, a4, v39, v40, v27, v33, v26);
  if ( (unsigned int)v42 > 0x40 )
  {
    if ( v41 )
    {
      v36 = result;
      j_j___libc_free_0_0(v41);
      return v36;
    }
  }
  return result;
}
