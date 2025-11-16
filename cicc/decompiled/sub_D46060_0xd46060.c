// Function: sub_D46060
// Address: 0xd46060
//
__int64 __fastcall sub_D46060(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 result; // rax
  __int64 v5; // r13
  __int64 v6; // rdx
  __int64 v7; // rdi
  unsigned int v8; // ecx
  __int64 v9; // rbx
  __int64 v10; // r8
  int v11; // r14d
  __int64 v12; // rax
  __int64 v13; // rax
  bool v14; // zf
  __int64 v15; // rdx
  int v16; // r10d
  __int64 v17; // [rsp+0h] [rbp-A0h] BYREF
  int v18; // [rsp+8h] [rbp-98h]
  void *v19; // [rsp+10h] [rbp-90h]
  unsigned __int64 v20[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v21; // [rsp+28h] [rbp-78h]
  __int64 v22; // [rsp+30h] [rbp-70h]
  void *v23; // [rsp+40h] [rbp-60h] BYREF
  _QWORD v24[2]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v25; // [rsp+58h] [rbp-48h]
  __int64 v26; // [rsp+60h] [rbp-40h]

  v3 = a1[1];
  v20[1] = 0;
  v20[0] = v3 & 6;
  v21 = a1[3];
  result = v21;
  if ( v21 != -4096 && v21 != 0 && v21 != -8192 )
  {
    sub_BD6050(v20, v3 & 0xFFFFFFFFFFFFFFF8LL);
    result = v21;
  }
  v5 = a1[4];
  v22 = v5;
  v19 = &unk_49DDFA0;
  v6 = *(unsigned int *)(v5 + 24);
  if ( (_DWORD)v6 )
  {
    v7 = *(_QWORD *)(v5 + 8);
    v8 = (v6 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
    v9 = v7 + 48LL * v8;
    v10 = *(_QWORD *)(v9 + 24);
    if ( result == v10 )
    {
LABEL_6:
      if ( v9 == v7 + 48 * v6 )
        goto LABEL_18;
      v11 = *(_DWORD *)(v9 + 40);
      v24[0] = 2;
      v24[1] = 0;
      v25 = -8192;
      v23 = &unk_49DDFA0;
      v26 = 0;
      v12 = *(_QWORD *)(v9 + 24);
      if ( v12 == -8192 )
      {
        *(_QWORD *)(v9 + 32) = 0;
      }
      else
      {
        if ( !v12 || v12 == -4096 )
        {
          *(_QWORD *)(v9 + 24) = -8192;
LABEL_13:
          v15 = v25;
          v14 = v25 == -4096;
          *(_QWORD *)(v9 + 32) = v26;
          v23 = &unk_49DB368;
          if ( v15 != -8192 && v15 != 0 && !v14 )
            sub_BD60C0(v24);
          goto LABEL_15;
        }
        sub_BD60C0((_QWORD *)(v9 + 8));
        v13 = v25;
        v14 = v25 == 0;
        *(_QWORD *)(v9 + 24) = v25;
        if ( v13 != -4096 && !v14 && v13 != -8192 )
        {
          sub_BD6050((unsigned __int64 *)(v9 + 8), v24[0] & 0xFFFFFFFFFFFFFFF8LL);
          goto LABEL_13;
        }
        *(_QWORD *)(v9 + 32) = v26;
      }
LABEL_15:
      --*(_DWORD *)(v5 + 16);
      ++*(_DWORD *)(v5 + 20);
      v17 = a2;
      v18 = v11;
      sub_D45B70((__int64)&v23, v22, &v17);
      result = v21;
      goto LABEL_18;
    }
    v16 = 1;
    while ( v10 != -4096 )
    {
      v8 = (v6 - 1) & (v16 + v8);
      v9 = v7 + 48LL * v8;
      v10 = *(_QWORD *)(v9 + 24);
      if ( result == v10 )
        goto LABEL_6;
      ++v16;
    }
  }
LABEL_18:
  v19 = &unk_49DB368;
  if ( result != 0 && result != -4096 && result != -8192 )
    return sub_BD60C0(v20);
  return result;
}
