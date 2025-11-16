// Function: sub_FDBA30
// Address: 0xfdba30
//
__int64 __fastcall sub_FDBA30(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rbx
  int v3; // edx
  int v4; // ecx
  __int64 v5; // rdi
  unsigned int v6; // edx
  _QWORD *v7; // r12
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rax
  int v11; // r8d
  _QWORD v12[2]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v13; // [rsp+10h] [rbp-40h]
  _QWORD v14[2]; // [rsp+20h] [rbp-30h] BYREF
  __int64 v15; // [rsp+30h] [rbp-20h]

  result = *(_QWORD *)(a1 + 24);
  v12[0] = 0;
  v12[1] = 0;
  v2 = *(_QWORD *)(a1 + 32);
  v13 = result;
  if ( result != 0 && result != -4096 && result != -8192 )
  {
    sub_BD73F0((__int64)v12);
    result = v13;
  }
  v3 = *(_DWORD *)(v2 + 184);
  if ( v3 )
  {
    v4 = v3 - 1;
    v5 = *(_QWORD *)(v2 + 168);
    v6 = (v3 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
    v7 = (_QWORD *)(v5 + 72LL * v6);
    v8 = v7[2];
    if ( v8 == result )
    {
LABEL_6:
      v7[4] = &unk_49DB368;
      v9 = v7[7];
      if ( v9 != 0 && v9 != -4096 && v9 != -8192 )
        sub_BD60C0(v7 + 5);
      v15 = -8192;
      v10 = v7[2];
      v14[0] = 0;
      v14[1] = 0;
      if ( v10 != -8192 )
      {
        if ( v10 && v10 != -4096 )
          sub_BD60C0(v7);
        v7[2] = -8192;
        if ( v15 != 0 && v15 != -4096 && v15 != -8192 )
          sub_BD60C0(v14);
      }
      --*(_DWORD *)(v2 + 176);
      result = v13;
      ++*(_DWORD *)(v2 + 180);
    }
    else
    {
      v11 = 1;
      while ( v8 != -4096 )
      {
        v6 = v4 & (v11 + v6);
        v7 = (_QWORD *)(v5 + 72LL * v6);
        v8 = v7[2];
        if ( v8 == result )
          goto LABEL_6;
        ++v11;
      }
    }
  }
  if ( result != 0 && result != -4096 && result != -8192 )
    return sub_BD60C0(v12);
  return result;
}
