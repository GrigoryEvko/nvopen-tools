// Function: sub_291C8F0
// Address: 0x291c8f0
//
__int64 __fastcall sub_291C8F0(__int64 a1, unsigned int **a2, unsigned __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  unsigned __int8 v7; // al
  unsigned __int8 v9; // cl
  unsigned int v10; // edx
  unsigned int v11; // r8d
  int v12; // edx
  __int64 **v13; // rax
  unsigned __int64 v14; // rax
  int v16; // edx
  char v17; // r9
  __int64 v18; // rcx
  char v19; // al
  __int64 v20; // rax
  int v21; // eax
  int v22; // r8d
  __int64 **v23; // rax
  __int64 **v24; // rax
  unsigned __int64 v25; // rax
  __int64 **v26; // rax
  int v27; // [rsp+8h] [rbp-88h]
  int v28[8]; // [rsp+10h] [rbp-80h] BYREF
  __int16 v29; // [rsp+30h] [rbp-60h]
  _BYTE v30[32]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v31; // [rsp+60h] [rbp-30h]

  v6 = *(_QWORD *)(a3 + 8);
  if ( a4 == v6 )
    return a3;
  v7 = *(_BYTE *)(v6 + 8);
  if ( v7 == 12 )
  {
    v9 = *(_BYTE *)(a4 + 8);
    if ( v9 == 12 )
    {
      v10 = *(_DWORD *)(a4 + 8) >> 8;
      if ( v10 > *(_DWORD *)(v6 + 8) >> 8 && v10 <= 8 )
      {
        v31 = 257;
        return sub_A82F30(a2, a3, a4, (__int64)v30, 0);
      }
    }
    v11 = -5;
    v12 = 12;
    goto LABEL_7;
  }
  v12 = v7;
  v11 = v7 - 17;
  if ( v11 > 1 )
    goto LABEL_17;
  if ( *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL) == 12 )
  {
    v9 = *(_BYTE *)(a4 + 8);
LABEL_7:
    if ( (unsigned int)v9 - 17 <= 1 )
      v9 = *(_BYTE *)(**(_QWORD **)(a4 + 16) + 8LL);
    if ( v9 == 14 )
    {
      v31 = 257;
      v29 = 257;
      v13 = (__int64 **)sub_AE4450(a1, a4);
      v14 = sub_291AC80((__int64 *)a2, 0x31u, a3, v13, (__int64)v28, 0, v27, 0);
      return sub_291AC80((__int64 *)a2, 0x30u, v14, (__int64 **)a4, (__int64)v30, 0, v27, 0);
    }
  }
  if ( v12 == 18 )
    goto LABEL_15;
LABEL_17:
  if ( v12 == 17 )
  {
LABEL_15:
    if ( *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL) != 14 )
    {
LABEL_16:
      v31 = 257;
      return sub_291AC80((__int64 *)a2, 0x31u, a3, (__int64 **)a4, (__int64)v30, 0, v28[0], 0);
    }
    goto LABEL_19;
  }
  if ( v7 != 14 )
    goto LABEL_16;
LABEL_19:
  v16 = *(unsigned __int8 *)(a4 + 8);
  v17 = *(_BYTE *)(a4 + 8);
  if ( (unsigned int)(v16 - 17) <= 1 )
    v17 = *(_BYTE *)(**(_QWORD **)(a4 + 16) + 8LL);
  if ( v17 == 12 )
  {
    v31 = 257;
    v29 = 257;
    v24 = (__int64 **)sub_AE4450(a1, v6);
    v25 = sub_291AC80((__int64 *)a2, 0x2Fu, a3, v24, (__int64)v28, 0, v27, 0);
    return sub_291AC80((__int64 *)a2, 0x31u, v25, (__int64 **)a4, (__int64)v30, 0, v27, 0);
  }
  else
  {
    if ( v11 <= 1 )
      v7 = *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL);
    if ( v7 != 14 )
      goto LABEL_16;
    v18 = a4;
    v19 = *(_BYTE *)(a4 + 8);
    if ( (unsigned int)(unsigned __int8)v16 - 17 <= 1 )
    {
      v18 = **(_QWORD **)(a4 + 16);
      v19 = *(_BYTE *)(v18 + 8);
    }
    if ( v19 != 14 )
      goto LABEL_16;
    v20 = v6;
    if ( v11 <= 1 )
      v20 = **(_QWORD **)(v6 + 16);
    v21 = *(_DWORD *)(v20 + 8) >> 8;
    v22 = *(_DWORD *)(v18 + 8) >> 8;
    if ( v22 == v21 )
      goto LABEL_16;
    v31 = 257;
    if ( v21 && v22 )
    {
      v29 = 257;
      v23 = (__int64 **)sub_AE4450(a1, v6);
      v14 = sub_291AC80((__int64 *)a2, 0x2Fu, a3, v23, (__int64)v28, 0, v27, 0);
      return sub_291AC80((__int64 *)a2, 0x30u, v14, (__int64 **)a4, (__int64)v30, 0, v27, 0);
    }
    v26 = (__int64 **)sub_BCE760((__int64 **)a4, v22);
    return sub_291AC80((__int64 *)a2, 0x32u, a3, v26, (__int64)v30, 0, v28[0], 0);
  }
}
