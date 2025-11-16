// Function: sub_1412B20
// Address: 0x1412b20
//
__int64 __fastcall sub_1412B20(__int64 a1, __int64 a2, char a3, _QWORD *a4, __int64 a5)
{
  _QWORD *v6; // r14
  int *v7; // rax
  __int64 v8; // r8
  int v9; // edx
  int v10; // eax
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rbx
  unsigned __int64 v13; // r12
  _QWORD *v14; // rdx
  char v15; // r9
  __int64 v16; // rsi
  unsigned __int8 v17; // al
  __int64 result; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned __int8 v22; // al
  unsigned __int64 v23; // rsi
  unsigned __int8 v24; // al
  __int64 v25; // rdx
  __int64 v26; // rsi
  unsigned __int8 v27; // al
  unsigned __int64 v28; // [rsp+0h] [rbp-80h]
  char v29; // [rsp+Ah] [rbp-76h]
  int v31; // [rsp+Ch] [rbp-74h]
  unsigned __int64 v32; // [rsp+10h] [rbp-70h]
  __m128i v34; // [rsp+20h] [rbp-60h] BYREF
  __int64 v35; // [rsp+30h] [rbp-50h]
  __int64 v36; // [rsp+38h] [rbp-48h]
  __int64 v37; // [rsp+40h] [rbp-40h]

  v6 = a4;
  v7 = (int *)sub_16D40F0(qword_4FBB410);
  if ( v7 )
    v9 = *v7;
  else
    v9 = qword_4FBB410[2];
  v10 = 5 * dword_4F994A0;
  if ( v9 > 2 )
    v10 = 32 * dword_4F994A0;
  v31 = v10;
  if ( a4 != *(_QWORD **)(a5 + 48) )
  {
    v32 = a2 & 0xFFFFFFFFFFFFFFF8LL;
    v28 = a2 & 0xFFFFFFFFFFFFFFF8LL | 4;
    while ( 1 )
    {
      v11 = *v6 & 0xFFFFFFFFFFFFFFF8LL;
      v12 = v11;
      v6 = (_QWORD *)v11;
      if ( !v11 )
        BUG();
      v13 = v11 - 24;
      if ( *(_BYTE *)(v11 - 8) == 78 )
      {
        v19 = *(_QWORD *)(v11 - 48);
        if ( !*(_BYTE *)(v19 + 16)
          && (*(_BYTE *)(v19 + 33) & 0x20) != 0
          && (unsigned int)(*(_DWORD *)(v19 + 36) - 35) <= 3 )
        {
          goto LABEL_21;
        }
      }
      if ( !--v31 )
        return 0x6000000000000003LL;
      v14 = *(_QWORD **)(a1 + 272);
      v34.m128i_i64[0] = 0;
      v34.m128i_i64[1] = -1;
      v35 = 0;
      v36 = 0;
      v37 = 0;
      v15 = sub_14113F0(v11 - 24, &v34, v14, v11, v8);
      if ( v34.m128i_i64[0] )
        break;
      v22 = *(_BYTE *)(v12 - 8);
      if ( v22 > 0x17u
        && ((v23 = v13, v22 == 78) || v22 == 29 && (v23 = v13 & 0xFFFFFFFFFFFFFFF8LL, (v13 & 0xFFFFFFFFFFFFFFF8LL) != 0)) )
      {
        v24 = *(_BYTE *)(v23 + 16);
        v25 = 0;
        if ( v24 > 0x17u )
        {
          if ( v24 == 78 )
          {
            v25 = v23 | 4;
          }
          else if ( v24 == 29 )
          {
            v25 = v23;
          }
        }
        v26 = 0;
        v27 = *(_BYTE *)(v32 + 16);
        if ( v27 > 0x17u )
        {
          if ( v27 == 78 )
          {
            v26 = v28;
          }
          else if ( v27 == 29 )
          {
            v26 = v32;
          }
        }
        v29 = v15;
        if ( (sub_134F530(*(_QWORD **)(a1 + 256), v26, v25) & 3) != 0 )
          return v13 | 1;
        if ( a3 && (v29 & 2) == 0 && (unsigned __int8)sub_15F40E0(v32, v13) )
          return v13 | 2;
LABEL_21:
        if ( *(_QWORD *)(a5 + 48) == v12 )
          goto LABEL_22;
      }
      else
      {
        if ( (v15 & 3) != 0 )
          return v13 | 1;
        if ( *(_QWORD *)(a5 + 48) == v12 )
          goto LABEL_22;
      }
    }
    v16 = 0;
    v17 = *(_BYTE *)(v32 + 16);
    if ( v17 > 0x17u )
    {
      if ( v17 == 78 )
      {
        v16 = v28;
      }
      else if ( v17 == 29 )
      {
        v16 = v32;
      }
    }
    if ( (sub_134F0E0(*(_QWORD **)(a1 + 256), v16, (__int64)&v34) & 3) != 0 )
      return v13 | 1;
    goto LABEL_21;
  }
LABEL_22:
  v20 = *(_QWORD *)(*(_QWORD *)(a5 + 56) + 80LL);
  if ( !v20 )
    return 0x2000000000000003LL;
  v21 = v20 - 24;
  result = 0x4000000000000003LL;
  if ( a5 != v21 )
    return 0x2000000000000003LL;
  return result;
}
