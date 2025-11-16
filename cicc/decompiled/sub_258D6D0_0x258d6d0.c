// Function: sub_258D6D0
// Address: 0x258d6d0
//
__int64 __fastcall sub_258D6D0(__int64 a1, __int64 a2)
{
  __int16 v4; // ax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rax
  unsigned int v12; // r12d
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdi
  char v17; // [rsp+7h] [rbp-189h] BYREF
  unsigned __int8 *v18; // [rsp+8h] [rbp-188h] BYREF
  unsigned __int64 v19; // [rsp+10h] [rbp-180h] BYREF
  __int64 v20; // [rsp+18h] [rbp-178h] BYREF
  __int64 v21[6]; // [rsp+20h] [rbp-170h] BYREF
  _QWORD v22[2]; // [rsp+50h] [rbp-140h] BYREF
  __int16 v23; // [rsp+60h] [rbp-130h]
  __int64 v24; // [rsp+68h] [rbp-128h]
  __int64 v25; // [rsp+70h] [rbp-120h]
  __int64 v26; // [rsp+78h] [rbp-118h]
  __int64 v27; // [rsp+80h] [rbp-110h]
  unsigned __int64 v28[2]; // [rsp+88h] [rbp-108h] BYREF
  _BYTE v29[248]; // [rsp+98h] [rbp-F8h] BYREF

  v24 = 0;
  v25 = 0;
  v26 = 0;
  v22[0] = &unk_4A171B8;
  v4 = *(_WORD *)(a1 + 104);
  v27 = 0;
  v23 = v4;
  v22[1] = &unk_4A16CD8;
  sub_C7D6A0(0, 0, 8);
  v9 = *(unsigned int *)(a1 + 136);
  LODWORD(v27) = v9;
  if ( (_DWORD)v9 )
  {
    v14 = sub_C7D670(24 * v9, 8);
    v6 = *(_QWORD *)(a1 + 120);
    v25 = v14;
    v5 = v14;
    v26 = *(_QWORD *)(a1 + 128);
    v15 = 0;
    v16 = 24LL * (unsigned int)v27;
    do
    {
      *(__m128i *)(v5 + v15) = _mm_loadu_si128((const __m128i *)(v6 + v15));
      *(_QWORD *)(v5 + v15 + 16) = *(_QWORD *)(v6 + v15 + 16);
      v15 += 24;
    }
    while ( v15 != v16 );
  }
  else
  {
    v25 = 0;
    v26 = 0;
  }
  v10 = *(unsigned int *)(a1 + 152);
  v28[0] = (unsigned __int64)v29;
  v28[1] = 0x800000000LL;
  if ( (_DWORD)v10 )
  {
    v10 = a1 + 144;
    sub_2539BB0((__int64)v28, a1 + 144, v5, v6, v7, v8);
  }
  v29[192] = *(_BYTE *)(a1 + 352);
  v18 = sub_250CBE0((__int64 *)(a1 + 72), v10);
  if ( !v18 )
    goto LABEL_8;
  v17 = 0;
  v19 = sub_2509740((_QWORD *)(a1 + 72));
  if ( sub_B49200(v19) )
  {
    sub_250D230((unsigned __int64 *)v21, v19, 1, 0);
    if ( !(unsigned __int8)sub_251C230(a2, v21, a1, 0, &v17, 0, 1) )
      goto LABEL_8;
  }
  v11 = sub_B491C0(v19);
  v21[0] = a2;
  v20 = v11;
  v21[1] = (__int64)&v18;
  v21[4] = (__int64)&v19;
  v21[2] = a1;
  v21[3] = (__int64)&v17;
  v21[5] = (__int64)&v20;
  if ( !(unsigned __int8)sub_258D3C0((__int64)v21, 1) )
    goto LABEL_8;
  if ( (unsigned __int8)sub_258D3C0((__int64)v21, 2) )
  {
    v12 = (unsigned __int8)sub_255BFA0((__int64)v22, (_BYTE *)(a1 + 88));
  }
  else
  {
LABEL_8:
    v12 = 0;
    *(_BYTE *)(a1 + 105) = *(_BYTE *)(a1 + 104);
  }
  v22[0] = &unk_4A171B8;
  if ( (_BYTE *)v28[0] != v29 )
    _libc_free(v28[0]);
  sub_C7D6A0(v25, 24LL * (unsigned int)v27, 8);
  return v12;
}
