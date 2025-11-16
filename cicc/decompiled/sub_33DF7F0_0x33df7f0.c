// Function: sub_33DF7F0
// Address: 0x33df7f0
//
__int64 __fastcall sub_33DF7F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  unsigned __int16 *v9; // rdx
  int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  unsigned int v14; // eax
  bool v15; // zf
  unsigned __int64 v16; // rdx
  unsigned int v17; // edi
  unsigned __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // [rsp+0h] [rbp-90h]
  __int64 v21; // [rsp+0h] [rbp-90h]
  int v22; // [rsp+8h] [rbp-88h]
  unsigned __int64 v23; // [rsp+20h] [rbp-70h] BYREF
  __int64 v24; // [rsp+28h] [rbp-68h]
  unsigned __int64 v25; // [rsp+30h] [rbp-60h]
  unsigned int v26; // [rsp+38h] [rbp-58h]
  unsigned __int64 v27; // [rsp+40h] [rbp-50h] BYREF
  __int64 v28; // [rsp+48h] [rbp-48h]
  unsigned __int64 v29; // [rsp+50h] [rbp-40h]
  unsigned int v30; // [rsp+58h] [rbp-38h]

  if ( sub_33CF170(a4) || sub_33CF4D0(a4) )
    return 0;
  v9 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  LOWORD(v23) = v10;
  v24 = v11;
  if ( (_WORD)v10 )
  {
    if ( (unsigned __int16)(v10 - 17) > 0xD3u )
    {
      LOWORD(v27) = v10;
      v28 = v11;
      goto LABEL_32;
    }
    LOWORD(v10) = word_4456580[v10 - 1];
    v19 = 0;
  }
  else
  {
    v20 = v11;
    if ( !sub_30070B0((__int64)&v23) )
    {
      v28 = v20;
      LOWORD(v27) = 0;
LABEL_8:
      LODWORD(v21) = sub_3007260((__int64)&v27);
      goto LABEL_9;
    }
    LOWORD(v10) = sub_3009970((__int64)&v23, a5, v20, v12, v13);
  }
  LOWORD(v27) = v10;
  v28 = v19;
  if ( !(_WORD)v10 )
    goto LABEL_8;
LABEL_32:
  if ( (_WORD)v10 == 1 || (unsigned __int16)(v10 - 504) <= 7u )
    BUG();
  v21 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v10 - 16];
LABEL_9:
  v22 = sub_33D4D80(a1, a2, a3, 0);
  v14 = v22 + sub_33D4D80(a1, a4, a5, 0);
  v15 = (_DWORD)v21 + 1 == v14;
  if ( (int)v21 + 1 < v14 )
    return 0;
  result = 1;
  if ( !v15 )
    return result;
  sub_33DD090((__int64)&v23, a1, a2, a3, 0);
  sub_33DD090((__int64)&v27, a1, a4, a5, 0);
  if ( (unsigned int)v24 > 0x40 )
    v16 = *(_QWORD *)(v23 + 8LL * ((unsigned int)(v24 - 1) >> 6));
  else
    v16 = v23;
  if ( (v16 & (1LL << ((unsigned __int8)v24 - 1))) != 0 )
    goto LABEL_40;
  v17 = v28;
  v18 = v27;
  if ( (unsigned int)v28 > 0x40 )
    v18 = *(_QWORD *)(v27 + 8LL * ((unsigned int)(v28 - 1) >> 6));
  if ( (v18 & (1LL << ((unsigned __int8)v28 - 1))) != 0 )
  {
LABEL_40:
    if ( v30 > 0x40 && v29 )
      j_j___libc_free_0_0(v29);
    if ( (unsigned int)v28 > 0x40 && v27 )
      j_j___libc_free_0_0(v27);
    if ( v26 > 0x40 && v25 )
      j_j___libc_free_0_0(v25);
    if ( (unsigned int)v24 > 0x40 && v23 )
      j_j___libc_free_0_0(v23);
    return 0;
  }
  if ( v30 > 0x40 && v29 )
  {
    j_j___libc_free_0_0(v29);
    v17 = v28;
  }
  if ( v17 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  if ( (unsigned int)v24 > 0x40 )
  {
    if ( v23 )
      j_j___libc_free_0_0(v23);
  }
  return 1;
}
