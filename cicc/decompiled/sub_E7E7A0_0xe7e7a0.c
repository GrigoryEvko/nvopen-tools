// Function: sub_E7E7A0
// Address: 0xe7e7a0
//
__int64 __fastcall sub_E7E7A0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // rdx
  __int64 v6; // r13
  char v7; // al
  __int64 v8; // rcx
  __int64 v9; // rdi
  int v10; // eax
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 result; // rax
  __int64 v15; // rdi
  __int64 *v16; // rax
  __int64 v17; // rsi
  _QWORD *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // [rsp-10h] [rbp-F0h]
  __m128i v22[2]; // [rsp+0h] [rbp-E0h] BYREF
  __int16 v23; // [rsp+20h] [rbp-C0h]
  __m128i v24; // [rsp+30h] [rbp-B0h] BYREF
  char *v25; // [rsp+40h] [rbp-A0h]
  __int16 v26; // [rsp+50h] [rbp-90h]
  __m128i v27; // [rsp+60h] [rbp-80h] BYREF
  _QWORD *v28; // [rsp+70h] [rbp-70h]
  __int64 v29; // [rsp+78h] [rbp-68h]
  __int16 v30; // [rsp+80h] [rbp-60h]
  __m128i *v31; // [rsp+90h] [rbp-50h] BYREF
  char v32; // [rsp+98h] [rbp-48h] BYREF
  char *v33; // [rsp+A0h] [rbp-40h]
  __int16 v34; // [rsp+B0h] [rbp-30h]
  char v35; // [rsp+B8h] [rbp-28h]

  v5 = *a2;
  v6 = *(_QWORD *)(*a2 + 16);
  v7 = *(_BYTE *)(v6 + 8);
  if ( (v7 & 2) == 0 )
    goto LABEL_5;
  v8 = *(_QWORD *)v6;
  if ( *(_QWORD *)v6 )
    goto LABEL_3;
  if ( (*(_BYTE *)(v6 + 9) & 0x70) == 0x20 && v7 >= 0 )
  {
    v19 = *(_QWORD *)(v6 + 24);
    *(_BYTE *)(v6 + 8) = v7 | 8;
    v20 = sub_E807D0(v19);
    *(_QWORD *)v6 = v20;
    v8 = v20;
    if ( !v20 )
    {
LABEL_13:
      v5 = *a2;
      v7 = *(_BYTE *)(v6 + 8);
      goto LABEL_8;
    }
LABEL_3:
    if ( (_UNKNOWN *)v8 != off_4C5D170 )
    {
      v9 = *(_QWORD *)(*(_QWORD *)(v8 + 8) + 16LL);
      *(_BYTE *)(v9 + 9) |= 8u;
      *a2 = sub_E808D0(v9, 0, *(_QWORD *)(a1 + 8), *(_QWORD *)(*a2 + 8));
LABEL_5:
      v10 = sub_E81A90(a3, *(_QWORD *)(a1 + 8), 0, 0);
      sub_E8BD00(
        (unsigned int)&v31,
        a1,
        v10,
        (unsigned int)"BFD_RELOC_NONE",
        14,
        *a2,
        *(_QWORD *)(*a2 + 8),
        *(_QWORD *)(*(_QWORD *)(a1 + 8) + 176LL));
      result = v21;
      if ( v35 )
      {
        v23 = 260;
        v22[0].m128i_i64[0] = (__int64)&v32;
        v24.m128i_i64[0] = (__int64)"Relocation for CG Profile could not be created: ";
        v26 = 259;
        sub_9C6370(&v27, &v24, v22, v11, v12, v13);
        sub_C64D30((__int64)&v27, 1u);
      }
      return result;
    }
    goto LABEL_13;
  }
LABEL_8:
  v15 = *(_QWORD *)(a1 + 8);
  if ( (v7 & 1) != 0 )
  {
    v16 = *(__int64 **)(v6 - 8);
    v17 = *v16;
    v18 = v16 + 3;
  }
  else
  {
    v17 = 0;
    v18 = 0;
  }
  v28 = v18;
  v26 = 771;
  v24.m128i_i64[0] = (__int64)"Reference to undefined temporary symbol ";
  v25 = "`";
  v27.m128i_i64[0] = (__int64)&v24;
  v29 = v17;
  v30 = 1282;
  v31 = &v27;
  v33 = "`";
  v34 = 770;
  return sub_E66880(v15, *(_QWORD **)(v5 + 8), (__int64)&v31);
}
