// Function: sub_147EA10
// Address: 0x147ea10
//
__int64 __fastcall sub_147EA10(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  __int64 v9; // rdx
  int v10; // eax
  __int64 v11; // rsi
  __int64 v12; // rdi
  int v13; // ecx
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // r8
  __int64 v17; // r14
  char v18; // al
  _QWORD *v19; // r12
  __int64 v20; // rdi
  __int64 v21; // rax
  unsigned int v23; // esi
  int v24; // eax
  int v25; // eax
  __int64 v26; // rdx
  __int64 v27; // rdx
  int v28; // eax
  int v29; // r9d
  __int64 v30; // [rsp+0h] [rbp-F0h]
  unsigned int v31; // [rsp+24h] [rbp-CCh]
  __int64 v32; // [rsp+28h] [rbp-C8h]
  void *v33; // [rsp+30h] [rbp-C0h] BYREF
  _QWORD v34[5]; // [rsp+38h] [rbp-B8h] BYREF
  _QWORD *v35; // [rsp+60h] [rbp-90h] BYREF
  _BYTE v36[16]; // [rsp+68h] [rbp-88h] BYREF
  __int64 v37; // [rsp+78h] [rbp-78h]
  int v38; // [rsp+90h] [rbp-60h] BYREF
  __int64 v39; // [rsp+98h] [rbp-58h]
  __int64 v40; // [rsp+A0h] [rbp-50h]
  char v41; // [rsp+A8h] [rbp-48h]
  char v42; // [rsp+A9h] [rbp-47h]
  char v43; // [rsp+B8h] [rbp-38h]

  v4 = 0;
  v9 = *(_QWORD *)(a1 + 64);
  v10 = *(_DWORD *)(v9 + 24);
  if ( v10 )
  {
    v11 = *(_QWORD *)(a2 + 40);
    v12 = *(_QWORD *)(v9 + 8);
    v13 = v10 - 1;
    v14 = (v10 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
    v15 = (__int64 *)(v12 + 16LL * v14);
    v16 = *v15;
    if ( v11 == *v15 )
    {
LABEL_3:
      v4 = v15[1];
    }
    else
    {
      v28 = 1;
      while ( v16 != -8 )
      {
        v29 = v28 + 1;
        v14 = v13 & (v28 + v14);
        v15 = (__int64 *)(v12 + 16LL * v14);
        v16 = *v15;
        if ( v11 == *v15 )
          goto LABEL_3;
        v28 = v29;
      }
      v4 = 0;
    }
  }
  sub_1455040((__int64)&v38, a3, *(_QWORD *)(a1 + 56));
  if ( !v43 || v38 != 11 )
    return 0;
  if ( v39 == a2 && sub_13FC1A0(v4, v40) )
  {
    v32 = sub_146F1B0(a1, v40);
  }
  else
  {
    if ( v40 != a2 || !sub_13FC1A0(v4, v39) )
      return 0;
    v32 = sub_146F1B0(a1, v39);
  }
  if ( !v32 )
    return 0;
  v31 = 2 * (v42 != 0);
  if ( v41 )
    v31 = (2 * (v42 != 0)) | 4;
  v30 = sub_146F1B0(a1, a4);
  v17 = sub_14799E0(a1, v30, v32, v4, v31);
  sub_1457D90(&v33, a2, a1);
  v18 = sub_145F6E0(a1 + 144, (__int64)&v33, &v35);
  v19 = v35;
  v20 = a1 + 144;
  if ( v18 )
    goto LABEL_14;
  v23 = *(_DWORD *)(a1 + 168);
  v24 = *(_DWORD *)(a1 + 160);
  ++*(_QWORD *)(a1 + 144);
  v25 = v24 + 1;
  if ( 4 * v25 >= 3 * v23 )
  {
    v23 *= 2;
LABEL_34:
    sub_14676C0(v20, v23);
    sub_145F6E0(v20, (__int64)&v33, &v35);
    v19 = v35;
    v25 = *(_DWORD *)(a1 + 160) + 1;
    goto LABEL_24;
  }
  if ( v23 - *(_DWORD *)(a1 + 164) - v25 <= v23 >> 3 )
    goto LABEL_34;
LABEL_24:
  *(_DWORD *)(a1 + 160) = v25;
  sub_1457D90(&v35, -8, 0);
  v26 = v37;
  if ( v37 != v19[3] )
    --*(_DWORD *)(a1 + 164);
  v35 = &unk_49EE2B0;
  if ( v26 != 0 && v26 != -8 && v26 != -16 )
    sub_1649B30(v36);
  sub_1453650((__int64)(v19 + 1), v34);
  v27 = v34[3];
  v19[5] = 0;
  v19[4] = v27;
LABEL_14:
  v19[5] = v17;
  v33 = &unk_49EE2B0;
  sub_1455FA0((__int64)v34);
  if ( *(_BYTE *)(a3 + 16) > 0x17u && sub_146CEE0(a1, v32, v4) && (unsigned __int8)sub_1471300(a1, a3, v4) )
  {
    v21 = sub_13A5B00(a1, v30, v32, 0, 0);
    sub_14799E0(a1, v21, v32, v4, v31);
  }
  return v17;
}
