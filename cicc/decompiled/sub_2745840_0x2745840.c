// Function: sub_2745840
// Address: 0x2745840
//
_QWORD *__fastcall sub_2745840(__int64 a1, __int64 a2)
{
  unsigned int v3; // esi
  __int64 v4; // rax
  _QWORD *v5; // r13
  int v6; // ecx
  __int64 v7; // rdx
  unsigned __int64 *v8; // r12
  __int64 v9; // rdx
  _QWORD *v10; // r12
  __int64 v11; // rdi
  unsigned int v12; // ecx
  _QWORD *v13; // r12
  __int64 v14; // rdx
  int v16; // r9d
  int v17; // edi
  _QWORD *v18; // [rsp+8h] [rbp-58h] BYREF
  void *v19; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v20[2]; // [rsp+18h] [rbp-48h] BYREF
  __int64 v21; // [rsp+28h] [rbp-38h]
  __int64 v22; // [rsp+30h] [rbp-30h]

  v20[0] = 2;
  v20[1] = 0;
  v21 = a2;
  if ( a2 != 0 && a2 != -4096 && a2 != -8192 )
    sub_BD73F0((__int64)v20);
  v3 = *(_DWORD *)(a1 + 24);
  v22 = a1;
  v19 = &unk_49DD7B0;
  if ( !v3 )
  {
    ++*(_QWORD *)a1;
    v18 = 0;
LABEL_6:
    v3 *= 2;
LABEL_7:
    sub_CF32C0(a1, v3);
    sub_F9E960(a1, (__int64)&v19, &v18);
    v4 = v21;
    v5 = v18;
    v6 = *(_DWORD *)(a1 + 16) + 1;
    goto LABEL_8;
  }
  v4 = v21;
  v11 = *(_QWORD *)(a1 + 8);
  v12 = (v3 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
  v13 = (_QWORD *)(v11 + ((unsigned __int64)v12 << 6));
  v14 = v13[3];
  if ( v14 == v21 )
  {
LABEL_19:
    v10 = v13 + 5;
    goto LABEL_20;
  }
  v16 = 1;
  v5 = 0;
  while ( v14 != -4096 )
  {
    if ( !v5 && v14 == -8192 )
      v5 = v13;
    v12 = (v3 - 1) & (v16 + v12);
    v13 = (_QWORD *)(v11 + ((unsigned __int64)v12 << 6));
    v14 = v13[3];
    if ( v21 == v14 )
      goto LABEL_19;
    ++v16;
  }
  v17 = *(_DWORD *)(a1 + 16);
  if ( !v5 )
    v5 = v13;
  ++*(_QWORD *)a1;
  v6 = v17 + 1;
  v18 = v5;
  if ( 4 * (v17 + 1) >= 3 * v3 )
    goto LABEL_6;
  if ( v3 - *(_DWORD *)(a1 + 20) - v6 <= v3 >> 3 )
    goto LABEL_7;
LABEL_8:
  *(_DWORD *)(a1 + 16) = v6;
  if ( v5[3] == -4096 )
  {
    v8 = v5 + 1;
    if ( v4 != -4096 )
    {
LABEL_13:
      v5[3] = v4;
      if ( v4 != -4096 && v4 != 0 && v4 != -8192 )
        sub_BD6050(v8, v20[0] & 0xFFFFFFFFFFFFFFF8LL);
      v4 = v21;
    }
  }
  else
  {
    --*(_DWORD *)(a1 + 20);
    v7 = v5[3];
    if ( v7 != v4 )
    {
      v8 = v5 + 1;
      if ( v7 != 0 && v7 != -4096 && v7 != -8192 )
      {
        sub_BD60C0(v5 + 1);
        v4 = v21;
      }
      goto LABEL_13;
    }
  }
  v9 = v22;
  v5[5] = 6;
  v10 = v5 + 5;
  v5[6] = 0;
  v5[4] = v9;
  v5[7] = 0;
LABEL_20:
  v19 = &unk_49DB368;
  if ( v4 != 0 && v4 != -4096 && v4 != -8192 )
    sub_BD60C0(v20);
  return v10;
}
