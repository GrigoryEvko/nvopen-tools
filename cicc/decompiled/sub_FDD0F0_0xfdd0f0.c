// Function: sub_FDD0F0
// Address: 0xfdd0f0
//
__int64 __fastcall sub_FDD0F0(__int64 a1, __int64 a2)
{
  int v2; // edx
  __int64 v3; // rax
  int v4; // ecx
  __int64 v5; // rdi
  unsigned int v6; // edx
  __int64 v7; // rbx
  __int64 v8; // rsi
  int v9; // edx
  __int64 v10; // rsi
  unsigned int v11; // r12d
  int v13; // r8d
  _QWORD v14[2]; // [rsp+0h] [rbp-70h] BYREF
  __int64 v15; // [rsp+10h] [rbp-60h]
  int v16; // [rsp+20h] [rbp-50h]
  void *v17; // [rsp+28h] [rbp-48h]
  unsigned __int64 v18[2]; // [rsp+30h] [rbp-40h] BYREF
  __int64 v19; // [rsp+40h] [rbp-30h]
  __int64 v20; // [rsp+48h] [rbp-28h]

  v14[0] = 0;
  v14[1] = 0;
  v15 = a2;
  if ( a2 != 0 && a2 != -4096 && a2 != -8192 )
    sub_BD73F0((__int64)v14);
  v2 = *(_DWORD *)(a1 + 184);
  v3 = v15;
  if ( !v2 )
  {
LABEL_15:
    v16 = -1;
    v11 = -1;
    goto LABEL_9;
  }
  v3 = v15;
  v4 = v2 - 1;
  v5 = *(_QWORD *)(a1 + 168);
  v6 = (v2 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
  v7 = v5 + 72LL * v6;
  v8 = *(_QWORD *)(v7 + 16);
  if ( v15 != v8 )
  {
    v13 = 1;
    while ( v8 != -4096 )
    {
      v6 = v4 & (v13 + v6);
      v7 = v5 + 72LL * v6;
      v8 = *(_QWORD *)(v7 + 16);
      if ( v15 == v8 )
        goto LABEL_6;
      ++v13;
    }
    goto LABEL_15;
  }
LABEL_6:
  v9 = *(_DWORD *)(v7 + 24);
  v10 = *(_QWORD *)(v7 + 40);
  v18[1] = 0;
  v16 = v9;
  v18[0] = v10 & 6;
  v19 = *(_QWORD *)(v7 + 56);
  if ( v19 != 0 && v19 != -4096 && v19 != -8192 )
  {
    sub_BD6050(v18, v10 & 0xFFFFFFFFFFFFFFF8LL);
    v11 = v16;
    v20 = *(_QWORD *)(v7 + 64);
    v17 = &unk_49DB368;
    if ( v19 != 0 && v19 != -8192 && v19 != -4096 )
      sub_BD60C0(v18);
    v3 = v15;
  }
  else
  {
    v11 = v16;
  }
LABEL_9:
  if ( v3 != -4096 && v3 != 0 && v3 != -8192 )
    sub_BD60C0(v14);
  return v11;
}
