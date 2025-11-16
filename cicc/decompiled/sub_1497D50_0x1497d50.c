// Function: sub_1497D50
// Address: 0x1497d50
//
__int64 __fastcall sub_1497D50(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // r14
  char v5; // al
  __int64 v6; // rbx
  __int64 v7; // r14
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rax
  bool v11; // zf
  __int64 v12; // rsi
  bool v13; // al
  __int64 result; // rax
  int v15; // [rsp+Ch] [rbp-A4h]
  int v16; // [rsp+Ch] [rbp-A4h]
  __int64 v17; // [rsp+10h] [rbp-A0h] BYREF
  int v18; // [rsp+18h] [rbp-98h]
  void *v19; // [rsp+20h] [rbp-90h] BYREF
  _QWORD v20[2]; // [rsp+28h] [rbp-88h] BYREF
  __int64 v21; // [rsp+38h] [rbp-78h]
  __int64 v22; // [rsp+40h] [rbp-70h]
  void *v23; // [rsp+50h] [rbp-60h] BYREF
  _QWORD v24[2]; // [rsp+58h] [rbp-58h] BYREF
  __int64 v25; // [rsp+68h] [rbp-48h]
  __int64 v26; // [rsp+70h] [rbp-40h]

  v3 = a1[1];
  v20[1] = 0;
  v20[0] = v3 & 6;
  v21 = a1[3];
  if ( v21 != 0 && v21 != -8 && v21 != -16 )
    sub_1649AC0(v20, v3 & 0xFFFFFFFFFFFFFFF8LL);
  v4 = a1[4];
  v22 = v4;
  v19 = &unk_49EC740;
  v5 = sub_14663D0(v4, (__int64)&v19, &v23);
  v6 = (__int64)v23;
  if ( !v5 )
    v6 = *(_QWORD *)(v4 + 8) + 48LL * *(unsigned int *)(v4 + 24);
  v7 = v22;
  if ( v6 != *(_QWORD *)(v22 + 8) + 48LL * *(unsigned int *)(v22 + 24) )
  {
    v8 = *(_DWORD *)(v6 + 40);
    v24[0] = 2;
    v24[1] = 0;
    v25 = -16;
    v23 = &unk_49EC740;
    v26 = 0;
    v9 = *(_QWORD *)(v6 + 24);
    if ( v9 == -16 )
    {
      *(_QWORD *)(v6 + 32) = 0;
    }
    else
    {
      if ( v9 == -8 || !v9 )
      {
        *(_QWORD *)(v6 + 24) = -16;
        v12 = v26;
        v13 = v25 != -8 && v25 != 0 && v25 != -16;
LABEL_13:
        *(_QWORD *)(v6 + 32) = v12;
        v23 = &unk_49EE2B0;
        if ( v13 )
        {
          v16 = v8;
          sub_1649B30(v24);
          v8 = v16;
        }
        goto LABEL_15;
      }
      v15 = v8;
      sub_1649B30(v6 + 8);
      v10 = v25;
      v8 = v15;
      v11 = v25 == -8;
      *(_QWORD *)(v6 + 24) = v25;
      if ( v10 != 0 && !v11 && v10 != -16 )
      {
        sub_1649AC0(v6 + 8, v24[0] & 0xFFFFFFFFFFFFFFF8LL);
        v12 = v26;
        v8 = v15;
        v13 = v25 != -8 && v25 != 0 && v25 != -16;
        goto LABEL_13;
      }
      *(_QWORD *)(v6 + 32) = v26;
    }
LABEL_15:
    --*(_DWORD *)(v7 + 16);
    ++*(_DWORD *)(v7 + 20);
    v18 = v8;
    v17 = a2;
    sub_14974D0((__int64)&v23, v22, (__int64)&v17);
  }
  v19 = &unk_49EE2B0;
  result = v21;
  if ( v21 != -8 && v21 != 0 && v21 != -16 )
    return sub_1649B30(v20);
  return result;
}
