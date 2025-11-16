// Function: sub_DD3750
// Address: 0xdd3750
//
__int64 __fastcall sub_DD3750(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rsi
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 *v14; // rax
  void *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r14
  __int64 *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // [rsp+10h] [rbp-150h]
  __int64 v21; // [rsp+10h] [rbp-150h]
  __int64 v22; // [rsp+10h] [rbp-150h]
  __int64 v23; // [rsp+10h] [rbp-150h]
  __int64 *v24; // [rsp+10h] [rbp-150h]
  __int64 v25; // [rsp+10h] [rbp-150h]
  __int64 v26; // [rsp+18h] [rbp-148h] BYREF
  __int64 *v27; // [rsp+28h] [rbp-138h] BYREF
  __int64 v28; // [rsp+30h] [rbp-130h]
  __int64 v29; // [rsp+38h] [rbp-128h]
  __int64 v30; // [rsp+40h] [rbp-120h] BYREF
  __int64 v31; // [rsp+48h] [rbp-118h]
  __int64 v32; // [rsp+50h] [rbp-110h]
  __int64 v33; // [rsp+58h] [rbp-108h] BYREF
  unsigned int v34; // [rsp+60h] [rbp-100h]
  char v35; // [rsp+98h] [rbp-C8h] BYREF
  _QWORD v36[2]; // [rsp+A0h] [rbp-C0h] BYREF
  int v37; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v38; // [rsp+B4h] [rbp-ACh]

  v26 = a2;
  if ( *(_BYTE *)(sub_D95540(a2) + 8) != 14 )
    return v26;
  v3 = (__int64)v36;
  v38 = v26;
  v36[0] = &v37;
  v36[1] = 0x2000000003LL;
  v37 = 14;
  v27 = 0;
  result = (__int64)sub_C65B40(a1 + 1032, (__int64)v36, (__int64 *)&v27, (__int64)off_49DEA80);
  if ( !result )
  {
    v4 = *(_QWORD *)(a1 + 8);
    v5 = sub_D95540(v26);
    if ( *(_BYTE *)(v5 + 8) == 14 )
    {
      v3 = *(_DWORD *)(v5 + 8) >> 8;
      if ( *((_BYTE *)sub_AE2980(v4, v3) + 16) )
        goto LABEL_9;
    }
    v6 = *(_QWORD *)(a1 + 8);
    v7 = sub_D95540(v26);
    v8 = sub_AE4450(v6, v7);
    v9 = sub_9208B0(*(_QWORD *)(a1 + 8), v8);
    v10 = *(_QWORD *)(a1 + 8);
    v31 = v11;
    v21 = v10;
    v30 = v9;
    v12 = sub_D95540(v26);
    v3 = sub_D97090(a1, v12);
    v28 = sub_9208B0(v21, v3);
    v29 = v13;
    if ( v28 != v30 || (_BYTE)v29 != (_BYTE)v31 )
    {
LABEL_9:
      result = sub_D970F0(a1);
    }
    else
    {
      v3 = v26;
      if ( *(_WORD *)(v26 + 24) == 15 )
      {
        if ( **(_BYTE **)(v26 - 8) == 20 )
        {
          v3 = v8;
          result = (__int64)sub_DA2C50(a1, v8, 0, 0);
        }
        else
        {
          v15 = sub_C65D30((__int64)v36, (unsigned __int64 *)(a1 + 1064));
          v23 = v16;
          v17 = (__int64)v15;
          v18 = (__int64 *)sub_A777F0(0x30u, (__int64 *)(a1 + 1064));
          if ( v18 )
          {
            v19 = v23;
            v24 = v18;
            sub_D96B50((__int64)v18, v17, v19, v26, v8);
            v18 = v24;
          }
          v25 = (__int64)v18;
          sub_C657C0((__int64 *)(a1 + 1032), v18, v27, (__int64)off_49DEA80);
          v3 = v25;
          sub_DAEE00(a1, v25, &v26, 1);
          result = v25;
        }
      }
      else
      {
        v30 = a1;
        v14 = &v33;
        v31 = 0;
        v32 = 1;
        do
        {
          *v14 = -4096;
          v14 += 2;
        }
        while ( v14 != (__int64 *)&v35 );
        result = sub_DD2D80(&v30, v3);
        if ( (v32 & 1) == 0 )
        {
          v22 = result;
          v3 = 16LL * v34;
          sub_C7D6A0(v33, v3, 8);
          result = v22;
        }
      }
    }
  }
  if ( (int *)v36[0] != &v37 )
  {
    v20 = result;
    _libc_free(v36[0], v3);
    return v20;
  }
  return result;
}
