// Function: sub_39B9110
// Address: 0x39b9110
//
__int64 __fastcall sub_39B9110(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 *v6; // r8
  __int64 v7; // rdx
  __int64 *v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // r13
  unsigned __int64 v12; // r12
  _BYTE *v13; // rdx
  int v14; // ecx
  _BYTE *v15; // r9
  __int64 *v16; // rax
  __int64 v17; // rcx
  unsigned int v18; // r12d
  char v19; // al
  __int64 v20; // rbx
  _BYTE *v21; // rsi
  char v22; // al
  __int64 v24; // [rsp+8h] [rbp-68h]
  __int64 v25; // [rsp+8h] [rbp-68h]
  _BYTE *v26; // [rsp+10h] [rbp-60h] BYREF
  __int64 v27; // [rsp+18h] [rbp-58h]
  _BYTE v28[80]; // [rsp+20h] [rbp-50h] BYREF

  v6 = a1;
  v7 = 3LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v8 = *(__int64 **)(a2 - 8);
    v9 = (__int64)&v8[v7];
  }
  else
  {
    v9 = a2;
    v8 = (__int64 *)(a2 - v7 * 8);
  }
  v10 = v9 - (_QWORD)v8;
  v26 = v28;
  v27 = 0x400000000LL;
  v11 = 0xAAAAAAAAAAAAAAABLL * (v10 >> 3);
  v12 = v11;
  if ( (unsigned __int64)v10 > 0x60 )
  {
    v25 = v10;
    sub_16CD150((__int64)&v26, v28, 0xAAAAAAAAAAAAAAABLL * (v10 >> 3), 8, (int)a1, a6);
    v15 = v26;
    v14 = v27;
    v10 = v25;
    v6 = a1;
    v13 = &v26[8 * (unsigned int)v27];
  }
  else
  {
    v13 = v28;
    v14 = 0;
    v15 = v28;
  }
  if ( v10 > 0 )
  {
    v16 = v8;
    do
    {
      v17 = *v16;
      v13 += 8;
      v16 += 3;
      *((_QWORD *)v13 - 1) = v17;
      --v12;
    }
    while ( v12 );
    v15 = v26;
    v14 = v27;
  }
  LODWORD(v27) = v14 + v11;
  v18 = 0;
  v24 = (__int64)v6;
  if ( (unsigned int)sub_39B8CC0(v6, a2, (__int64)v15, (unsigned int)(v14 + v11)) )
  {
    v19 = *(_BYTE *)(a2 + 16);
    if ( v19 == 54 )
    {
      v18 = 4;
      goto LABEL_20;
    }
    v20 = *(_QWORD *)a2;
    if ( v19 == 78 )
    {
      v21 = *(_BYTE **)(a2 - 24);
      if ( v21[16] || sub_14A2090(v24, v21) )
      {
        v18 = 40;
        goto LABEL_20;
      }
      v22 = *(_BYTE *)(v20 + 8);
      if ( v22 != 13 )
        goto LABEL_17;
      v20 = **(_QWORD **)(v20 + 16);
    }
    v22 = *(_BYTE *)(v20 + 8);
LABEL_17:
    if ( v22 == 16 )
      v22 = *(_BYTE *)(*(_QWORD *)(v20 + 24) + 8LL);
    v18 = (unsigned __int8)(v22 - 1) < 6u ? 3 : 1;
  }
LABEL_20:
  if ( v26 != v28 )
    _libc_free((unsigned __int64)v26);
  return v18;
}
