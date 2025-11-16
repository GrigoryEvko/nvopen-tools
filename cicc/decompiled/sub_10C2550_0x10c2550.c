// Function: sub_10C2550
// Address: 0x10c2550
//
bool __fastcall sub_10C2550(int *a1, unsigned __int8 *a2, __int64 a3, __int64 a4, _BYTE *a5, _QWORD *a6)
{
  __int64 v6; // rax
  int v9; // eax
  int v10; // esi
  int v11; // edx
  _BYTE *v12; // r12
  _BYTE *v13; // r12
  char v14; // al
  unsigned __int8 *v15; // rsi
  unsigned __int8 *v16; // rsi
  __int64 v17; // r9
  __int64 v18; // rax
  char v19; // al
  unsigned __int8 *v20; // rsi
  unsigned __int8 *v21; // rsi
  _BYTE *v22; // r10
  _QWORD *v24; // [rsp-60h] [rbp-60h] BYREF
  __int64 v25[2]; // [rsp-58h] [rbp-58h] BYREF
  int v26; // [rsp-48h] [rbp-48h]
  _BYTE *v27; // [rsp-40h] [rbp-40h]
  int v28; // [rsp-38h] [rbp-38h]

  v6 = *((_QWORD *)a2 + 2);
  if ( !v6 || *(_QWORD *)(v6 + 8) )
    return 0;
  v9 = a1[1];
  v10 = *a1;
  v25[0] = a3;
  v11 = *a2;
  v28 = v9;
  v24 = 0;
  v25[1] = a4;
  v26 = v10;
  v27 = a5;
  if ( v11 != v9 + 29 )
    return 0;
  v12 = (_BYTE *)*((_QWORD *)a2 - 8);
  if ( !v12 )
    goto LABEL_8;
  *a6 = v12;
  if ( *v12 != 59 )
    goto LABEL_8;
  v19 = sub_995B10(&v24, *((_QWORD *)v12 - 8));
  v20 = (unsigned __int8 *)*((_QWORD *)v12 - 4);
  if ( v19 && *v20 == v26 + 29 && (unsigned __int8)sub_10B8260(v25, (__int64)v20) )
  {
    v22 = (_BYTE *)*((_QWORD *)a2 - 4);
    goto LABEL_24;
  }
  if ( !(unsigned __int8)sub_995B10(&v24, (__int64)v20) )
  {
LABEL_8:
    v13 = (_BYTE *)*((_QWORD *)a2 - 4);
    goto LABEL_9;
  }
  v21 = (unsigned __int8 *)*((_QWORD *)v12 - 8);
  v13 = (_BYTE *)*((_QWORD *)a2 - 4);
  if ( *v21 == v26 + 29 && (unsigned __int8)sub_10B8260(v25, (__int64)v21) )
  {
LABEL_24:
    v13 = v22;
    if ( v22 == v27 )
      goto LABEL_17;
  }
LABEL_9:
  if ( !v13 )
    return 0;
  *a6 = v13;
  if ( *v13 != 59 )
    return 0;
  v14 = sub_995B10(&v24, *((_QWORD *)v13 - 8));
  v15 = (unsigned __int8 *)*((_QWORD *)v13 - 4);
  if ( !v14 || *v15 != v26 + 29 || !(unsigned __int8)sub_10B8260(v25, (__int64)v15) )
  {
    if ( !(unsigned __int8)sub_995B10(&v24, (__int64)v15) )
      return 0;
    v16 = (unsigned __int8 *)*((_QWORD *)v13 - 8);
    if ( *v16 != v26 + 29 || !(unsigned __int8)sub_10B8260(v25, (__int64)v16) )
      return 0;
  }
  if ( *((_BYTE **)a2 - 8) != v27 )
    return 0;
LABEL_17:
  v18 = *(_QWORD *)(*(_QWORD *)v17 + 16LL);
  if ( !v18 )
    return 0;
  return *(_QWORD *)(v18 + 8) == 0;
}
