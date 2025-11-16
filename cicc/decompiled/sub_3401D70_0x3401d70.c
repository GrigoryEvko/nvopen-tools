// Function: sub_3401D70
// Address: 0x3401d70
//
unsigned __int8 *__fastcall sub_3401D70(unsigned __int16 **a1, __int64 a2, unsigned int a3, __int64 a4, __m128i a5)
{
  unsigned __int16 *v9; // rdi
  int v10; // r15d
  __int64 v11; // rdx
  int v12; // eax
  unsigned int v13; // edx
  __int64 v14; // r15
  unsigned int v15; // esi
  __int64 v16; // rax
  unsigned __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int16 v27; // [rsp+10h] [rbp-50h] BYREF
  __int64 v28; // [rsp+18h] [rbp-48h]

  v9 = *a1;
  v10 = *v9;
  if ( !(_WORD)v10 )
  {
    if ( sub_30070B0((__int64)v9) )
    {
      LOWORD(v10) = sub_3009970((__int64)v9, a2, v24, v25, v26);
LABEL_4:
      v27 = v10;
      v28 = v11;
      if ( !(_WORD)v10 )
        goto LABEL_5;
LABEL_19:
      if ( (_WORD)v10 == 1 || (unsigned __int16)(v10 - 504) <= 7u )
        BUG();
      v14 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v10 - 16];
      v13 = *(_DWORD *)(a2 + 8);
      v15 = v13 - v14;
      if ( v13 <= 0x40 )
        goto LABEL_6;
LABEL_22:
      sub_C47690((__int64 *)a2, v15);
      v13 = *(_DWORD *)(a2 + 8);
      v15 = v13 - v14;
      if ( v13 > 0x40 )
      {
        sub_C44B70(a2, v15);
        return sub_34007B0((__int64)a1[2], a2, (__int64)a1[1], a3, a4, 0, a5, 0);
      }
      v18 = *(_QWORD *)a2;
      v17 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v13;
      goto LABEL_10;
    }
LABEL_3:
    v11 = *((_QWORD *)v9 + 1);
    goto LABEL_4;
  }
  if ( (unsigned __int16)(v10 - 17) > 0xD3u )
    goto LABEL_3;
  v28 = 0;
  LOWORD(v10) = word_4456580[v10 - 1];
  v27 = v10;
  if ( (_WORD)v10 )
    goto LABEL_19;
LABEL_5:
  v12 = sub_3007260((__int64)&v27);
  v13 = *(_DWORD *)(a2 + 8);
  LODWORD(v14) = v12;
  v15 = v13 - v12;
  if ( v13 > 0x40 )
    goto LABEL_22;
LABEL_6:
  v16 = 0;
  if ( v15 != v13 )
    v16 = *(_QWORD *)a2 << v15;
  v17 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v13;
  if ( !v13 )
    goto LABEL_26;
  v18 = v17 & v16;
LABEL_10:
  if ( !v13 )
  {
LABEL_26:
    v19 = 0;
    goto LABEL_12;
  }
  v19 = v18 << (64 - (unsigned __int8)v13) >> (64 - (unsigned __int8)v13);
LABEL_12:
  v20 = v19 >> 63;
  v21 = v19 >> v15;
  if ( v13 == v15 )
    v21 = v20;
  v22 = v17 & v21;
  if ( !v13 )
    v22 = 0;
  *(_QWORD *)a2 = v22;
  return sub_34007B0((__int64)a1[2], a2, (__int64)a1[1], a3, a4, 0, a5, 0);
}
