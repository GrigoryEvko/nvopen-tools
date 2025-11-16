// Function: sub_14B2890
// Address: 0x14b2890
//
unsigned __int64 __fastcall sub_14B2890(__int64 a1, __int64 *a2, __int64 *a3, unsigned int *a4, unsigned int a5)
{
  unsigned int *v10; // rax
  unsigned int v11; // eax
  __int64 v13; // rdi
  char v14; // al
  char v15; // dl
  _QWORD *v16; // rsi
  __int64 v17; // r12
  unsigned int v18; // r10d
  __int64 v19; // rax
  unsigned int v20; // edx
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rsi
  unsigned int v23; // edx
  __int64 v24; // rax
  unsigned int v25; // [rsp+4h] [rbp-8Ch]
  __int64 v26; // [rsp+10h] [rbp-80h]
  __int16 v27; // [rsp+18h] [rbp-78h]
  _QWORD *v28; // [rsp+20h] [rbp-70h]

  v10 = (unsigned int *)sub_16D40F0(qword_4FBB370);
  if ( v10 )
    v11 = *v10;
  else
    v11 = qword_4FBB370[2];
  if ( a5 < v11 && *(_BYTE *)(a1 + 16) == 79 )
  {
    v13 = *(_QWORD *)(a1 - 72);
    v14 = *(_BYTE *)(v13 + 16);
    if ( (unsigned __int8)(v14 - 75) > 1u )
      goto LABEL_22;
    v15 = *(_BYTE *)(*(_QWORD *)v13 + 8LL);
    if ( v15 == 16 )
      v15 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)v13 + 16LL) + 8LL);
    v27 = *(_WORD *)(v13 + 18);
    v28 = *(_QWORD **)(v13 - 48);
    v16 = *(_QWORD **)(a1 - 48);
    v26 = *(_QWORD *)(v13 - 24);
    v17 = *(_QWORD *)(a1 - 24);
    if ( (unsigned __int8)(v15 - 1) <= 5u || v14 == 76 )
      v25 = sub_15F24E0(v13);
    else
      v25 = 0;
    if ( (unsigned __int8)sub_15FF0A0() )
    {
LABEL_22:
      v23 = 0;
      LODWORD(v22) = 0;
      return __PAIR64__(v23, v22);
    }
    v18 = v27 & 0x7FFF;
    if ( a4 && *v28 != *v16 )
    {
      v19 = sub_14A8D80(v13, (__int64)v16, v17, a4);
      if ( v19 )
      {
        v20 = v25 | 8;
        if ( *a4 - 39 >= 2 )
          v20 = v25;
        v21 = sub_14B0C10(v27 & 0x7FFF, v20, (__int64)v28, v26, *(v16 - 3), v19, a2, a3, a5);
        goto LABEL_21;
      }
      v24 = sub_14A8D80(v13, v17, (__int64)v16, a4);
      v18 = v27 & 0x7FFF;
      if ( v24 )
      {
        if ( *a4 - 39 <= 1 )
          v25 |= 8u;
        v22 = sub_14B0C10(v27 & 0x7FFF, v25, (__int64)v28, v26, v24, *(_QWORD *)(v17 - 24), a2, a3, a5);
        v23 = HIDWORD(v22);
        return __PAIR64__(v23, v22);
      }
    }
    v21 = sub_14B0C10(v18, v25, (__int64)v28, v26, (__int64)v16, v17, a2, a3, a5);
LABEL_21:
    LODWORD(v22) = v21;
    v23 = HIDWORD(v21);
    return __PAIR64__(v23, v22);
  }
  return 0;
}
