// Function: sub_2839560
// Address: 0x2839560
//
__int64 __fastcall sub_2839560(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 *v6; // r14
  __int64 v7; // r13
  __int64 v8; // rax
  char v9; // dl
  __int64 v10; // rax
  unsigned __int8 v11; // dl
  unsigned int v12; // r12d
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r14
  __int64 v17; // rax
  _QWORD *v18; // rax
  __int64 v19; // r14
  unsigned int v20; // ebx
  __int64 v21; // rax
  char v22; // [rsp+7h] [rbp-79h]
  __int64 v23; // [rsp+8h] [rbp-78h]
  __int64 v24; // [rsp+10h] [rbp-70h]
  __int64 v25; // [rsp+18h] [rbp-68h]
  __int64 v26; // [rsp+20h] [rbp-60h]
  unsigned __int64 v27; // [rsp+30h] [rbp-50h] BYREF
  __int64 v28; // [rsp+38h] [rbp-48h]
  __int64 v29; // [rsp+40h] [rbp-40h]
  unsigned int v30; // [rsp+48h] [rbp-38h]

  v25 = *(_QWORD *)(a1 - 32);
  v24 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)a1 == 61 )
    v6 = *(unsigned __int8 **)(a1 + 8);
  else
    v6 = *(unsigned __int8 **)(*(_QWORD *)(a1 - 64) + 8LL);
  v23 = sub_B43CC0(a1);
  v7 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v8 = sub_D34EB0(a3, v6, v25, a4, (__int64)&v27, 0, 1);
  if ( v9 )
    v7 = v8;
  sub_C7D6A0(v28, 16LL * v30, 8);
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v10 = sub_D34EB0(a3, v6, v24, a4, (__int64)&v27, 0, 1);
  v12 = v11;
  v26 = v10;
  if ( v11 )
  {
    sub_C7D6A0(v28, 16LL * v30, 8);
    if ( v26 != 0 && v7 != 0 && v26 == v7 && abs64(v7) == 1 )
    {
      v22 = sub_AE5020(v23, (__int64)v6);
      v14 = sub_9208B0(v23, (__int64)v6);
      v28 = v15;
      v27 = ((1LL << v22) + ((unsigned __int64)(v14 + 7) >> 3) - 1) >> v22 << v22;
      v12 = sub_CA1930(&v27);
      v16 = sub_DEEF40(a3, v25);
      v17 = sub_DEEF40(a3, v24);
      v18 = sub_DCC810(*(__int64 **)(a3 + 112), v17, v16, 0, 0);
      if ( !*((_WORD *)v18 + 12) )
      {
        v19 = v18[4];
        v20 = *(_DWORD *)(v19 + 32);
        if ( v20 <= 0x40 )
        {
          v21 = *(_QWORD *)(v19 + 24);
          goto LABEL_16;
        }
        if ( v20 - (unsigned int)sub_C444A0(v19 + 24) <= 0x40 )
        {
          v21 = **(_QWORD **)(v19 + 24);
LABEL_16:
          LOBYTE(v12) = v12 * v7 == v21;
          return v12;
        }
      }
    }
    return 0;
  }
  sub_C7D6A0(v28, 16LL * v30, 8);
  return v12;
}
