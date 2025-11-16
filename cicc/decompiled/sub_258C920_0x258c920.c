// Function: sub_258C920
// Address: 0x258c920
//
__int64 __fastcall sub_258C920(__int64 a1, __int64 a2)
{
  __int16 v3; // ax
  __int64 v4; // rdx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rax
  __int64 v8; // rcx
  _QWORD *v9; // r15
  unsigned __int64 *v10; // rbx
  char v11; // r11
  __int64 v12; // rax
  unsigned __int8 *v13; // r8
  char v14; // r9
  unsigned __int64 v15; // rcx
  __int64 v16; // rax
  unsigned int v17; // r12d
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // [rsp-10h] [rbp-200h]
  unsigned __int8 *v24; // [rsp+18h] [rbp-1D8h]
  unsigned __int64 *v25; // [rsp+20h] [rbp-1D0h]
  char v26; // [rsp+2Fh] [rbp-1C1h]
  char v27; // [rsp+3Bh] [rbp-1B5h] BYREF
  int v28; // [rsp+3Ch] [rbp-1B4h] BYREF
  _QWORD v29[6]; // [rsp+40h] [rbp-1B0h] BYREF
  unsigned __int64 *v30; // [rsp+70h] [rbp-180h] BYREF
  __int64 v31; // [rsp+78h] [rbp-178h]
  _BYTE v32[48]; // [rsp+80h] [rbp-170h] BYREF
  _QWORD v33[2]; // [rsp+B0h] [rbp-140h] BYREF
  __int16 v34; // [rsp+C0h] [rbp-130h]
  __int64 v35; // [rsp+C8h] [rbp-128h]
  __int64 v36; // [rsp+D0h] [rbp-120h]
  __int64 v37; // [rsp+D8h] [rbp-118h]
  __int64 v38; // [rsp+E0h] [rbp-110h]
  unsigned __int64 v39[2]; // [rsp+E8h] [rbp-108h] BYREF
  _BYTE v40[248]; // [rsp+F8h] [rbp-F8h] BYREF

  v35 = 0;
  v36 = 0;
  v37 = 0;
  v33[0] = &unk_4A171B8;
  v3 = *(_WORD *)(a1 + 104);
  v38 = 0;
  v34 = v3;
  v33[1] = &unk_4A16CD8;
  sub_C7D6A0(0, 0, 8);
  v7 = *(unsigned int *)(a1 + 136);
  LODWORD(v38) = v7;
  if ( (_DWORD)v7 )
  {
    v19 = sub_C7D670(24 * v7, 8);
    v20 = *(_QWORD *)(a1 + 120);
    v36 = v19;
    v4 = v19;
    v37 = *(_QWORD *)(a1 + 128);
    v21 = 0;
    v22 = 24LL * (unsigned int)v38;
    do
    {
      *(__m128i *)(v4 + v21) = _mm_loadu_si128((const __m128i *)(v20 + v21));
      *(_QWORD *)(v4 + v21 + 16) = *(_QWORD *)(v20 + v21 + 16);
      v21 += 24;
    }
    while ( v21 != v22 );
  }
  else
  {
    v36 = 0;
    v37 = 0;
  }
  v8 = *(unsigned int *)(a1 + 152);
  v39[0] = (unsigned __int64)v40;
  v39[1] = 0x800000000LL;
  if ( (_DWORD)v8 )
    sub_2539BB0((__int64)v39, a1 + 144, v4, v8, v5, v6);
  v40[192] = *(_BYTE *)(a1 + 352);
  v30 = (unsigned __int64 *)v32;
  v28 = sub_250CB50((__int64 *)(a1 + 72), 1);
  v31 = 0x300000000LL;
  v29[0] = &v28;
  v27 = 0;
  v29[1] = a2;
  v29[2] = a1;
  v29[3] = &v30;
  v29[4] = &v27;
  if ( (unsigned __int8)sub_2523890(
                          a2,
                          (__int64 (__fastcall *)(__int64, __int64 *))sub_254DEA0,
                          (__int64)v29,
                          a1,
                          1u,
                          &v27) )
  {
    v24 = sub_250CBE0((__int64 *)(a1 + 72), (__int64)sub_254DEA0);
    v25 = &v30[2 * (unsigned int)v31];
    if ( v30 != v25 )
    {
      v9 = (_QWORD *)(a1 + 72);
      v26 = 0;
      v10 = v30;
      do
      {
        if ( *(_BYTE *)*v10 <= 0x15u )
          goto LABEL_14;
        v11 = sub_252BB70(a2, a1, *v10, 1);
        if ( !v11 )
          goto LABEL_15;
        if ( *(_BYTE *)*v10 == 22 && v24 == *(unsigned __int8 **)(*v10 + 24) )
        {
LABEL_14:
          v16 = sub_25096F0(v9);
          v13 = (unsigned __int8 *)v10[1];
          v14 = 3;
          v23 = v16;
        }
        else
        {
          v26 = v11;
          v12 = sub_25096F0(v9);
          v13 = (unsigned __int8 *)v10[1];
          v14 = 2;
          v23 = v12;
        }
        v15 = *v10;
        v10 += 2;
        sub_258BA20(a1, a2, (_BYTE *)(a1 + 88), v15, v13, v14, v23);
      }
      while ( v25 != v10 );
      if ( v26 )
        sub_258C650(a1, a2);
    }
    v17 = (unsigned __int8)sub_255BFA0((__int64)v33, (_BYTE *)(a1 + 88));
  }
  else
  {
LABEL_15:
    v17 = sub_2579F40(a1);
  }
  if ( v30 != (unsigned __int64 *)v32 )
    _libc_free((unsigned __int64)v30);
  v33[0] = &unk_4A171B8;
  if ( (_BYTE *)v39[0] != v40 )
    _libc_free(v39[0]);
  sub_C7D6A0(v36, 24LL * (unsigned int)v38, 8);
  return v17;
}
