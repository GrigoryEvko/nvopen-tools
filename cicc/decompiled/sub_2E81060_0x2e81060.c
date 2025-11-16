// Function: sub_2E81060
// Address: 0x2e81060
//
__int64 __fastcall sub_2E81060(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 result; // rax
  __int64 v5; // r13
  __int64 v6; // rdx
  __int64 v7; // rdi
  unsigned int v8; // ecx
  _QWORD *v9; // rbx
  __int64 v10; // r8
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rax
  bool v14; // zf
  __int64 v15; // rcx
  bool v16; // al
  int v17; // r10d
  __int64 v18; // [rsp+0h] [rbp-A0h] BYREF
  __int64 v19; // [rsp+8h] [rbp-98h]
  void *v20; // [rsp+10h] [rbp-90h]
  unsigned __int64 v21[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v22; // [rsp+28h] [rbp-78h]
  __int64 v23; // [rsp+30h] [rbp-70h]
  void *v24; // [rsp+40h] [rbp-60h] BYREF
  _QWORD v25[2]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v26; // [rsp+58h] [rbp-48h]
  __int64 v27; // [rsp+60h] [rbp-40h]

  v3 = a1[1];
  v21[1] = 0;
  v21[0] = v3 & 6;
  v22 = a1[3];
  result = v22;
  if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
  {
    sub_BD6050(v21, v3 & 0xFFFFFFFFFFFFFFF8LL);
    result = v22;
  }
  v5 = a1[4];
  v23 = v5;
  v20 = &unk_4A28E90;
  v6 = *(unsigned int *)(v5 + 24);
  if ( (_DWORD)v6 )
  {
    v7 = *(_QWORD *)(v5 + 8);
    v8 = (v6 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
    v9 = (_QWORD *)(v7 + 48LL * v8);
    v10 = v9[3];
    if ( result == v10 )
    {
LABEL_6:
      if ( v9 == (_QWORD *)(v7 + 48 * v6) )
        goto LABEL_20;
      v11 = v9[5];
      v25[0] = 2;
      v25[1] = 0;
      v26 = -8192;
      v24 = &unk_4A28E90;
      v27 = 0;
      v12 = v9[3];
      if ( v12 == -8192 )
      {
        v9[4] = 0;
      }
      else
      {
        if ( !v12 || v12 == -4096 )
        {
          v9[3] = -8192;
          v15 = v27;
          v16 = v26 != -8192 && v26 != 0 && v26 != -4096;
LABEL_13:
          v9[4] = v15;
          v24 = &unk_49DB368;
          if ( v16 )
            sub_BD60C0(v25);
          goto LABEL_15;
        }
        sub_BD60C0(v9 + 1);
        v13 = v26;
        v14 = v26 == 0;
        v9[3] = v26;
        if ( v13 != -4096 && !v14 && v13 != -8192 )
        {
          sub_BD6050(v9 + 1, v25[0] & 0xFFFFFFFFFFFFFFF8LL);
          v15 = v27;
          v16 = v26 != -4096 && v26 != -8192 && v26 != 0;
          goto LABEL_13;
        }
        v9[4] = v27;
      }
LABEL_15:
      --*(_DWORD *)(v5 + 16);
      ++*(_DWORD *)(v5 + 20);
      v18 = a2;
      v19 = v11;
      sub_2E80B70((__int64)&v24, v23, &v18);
      if ( v19 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v19 + 16LL))(v19);
      result = v22;
      goto LABEL_20;
    }
    v17 = 1;
    while ( v10 != -4096 )
    {
      v8 = (v6 - 1) & (v17 + v8);
      v9 = (_QWORD *)(v7 + 48LL * v8);
      v10 = v9[3];
      if ( v10 == result )
        goto LABEL_6;
      ++v17;
    }
  }
LABEL_20:
  v20 = &unk_49DB368;
  if ( result != 0 && result != -4096 && result != -8192 )
    return sub_BD60C0(v21);
  return result;
}
