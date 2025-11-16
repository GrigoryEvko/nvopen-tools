// Function: sub_1466490
// Address: 0x1466490
//
__int64 __fastcall sub_1466490(_QWORD *a1)
{
  __int64 v1; // rsi
  __int64 v2; // rbx
  char v3; // al
  _QWORD *v4; // r13
  __int64 result; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  bool v8; // zf
  __int64 v9; // rdx
  bool v10; // al
  void *v11; // [rsp+0h] [rbp-80h] BYREF
  _QWORD v12[2]; // [rsp+8h] [rbp-78h] BYREF
  __int64 v13; // [rsp+18h] [rbp-68h]
  __int64 v14; // [rsp+20h] [rbp-60h]
  _QWORD *v15; // [rsp+30h] [rbp-50h] BYREF
  _QWORD v16[2]; // [rsp+38h] [rbp-48h] BYREF
  __int64 v17; // [rsp+48h] [rbp-38h]
  __int64 v18; // [rsp+50h] [rbp-30h]

  v1 = a1[1];
  v12[1] = 0;
  v12[0] = v1 & 6;
  v13 = a1[3];
  if ( v13 != -8 && v13 != 0 && v13 != -16 )
    sub_1649AC0(v12, v1 & 0xFFFFFFFFFFFFFFF8LL);
  v2 = a1[4];
  v14 = v2;
  v11 = &unk_49EC740;
  v3 = sub_14663D0(v2, (__int64)&v11, &v15);
  v4 = v15;
  if ( v3 )
  {
    v16[0] = 2;
    v16[1] = 0;
    v17 = -16;
    v15 = &unk_49EC740;
    v18 = 0;
    v6 = v4[3];
    if ( v6 == -16 )
    {
      v4[4] = 0;
    }
    else
    {
      if ( v6 == -8 || !v6 )
      {
        v4[3] = -16;
LABEL_15:
        v9 = v17;
        v10 = v17 != -8;
        v8 = v17 == 0;
        v4[4] = v18;
        v15 = &unk_49EE2B0;
        if ( v9 != -16 && !v8 && v10 )
          sub_1649B30(v16);
        goto LABEL_17;
      }
      sub_1649B30(v4 + 1);
      v7 = v17;
      v8 = v17 == -8;
      v4[3] = v17;
      if ( v7 != 0 && !v8 && v7 != -16 )
      {
        sub_1649AC0(v4 + 1, v16[0] & 0xFFFFFFFFFFFFFFF8LL);
        goto LABEL_15;
      }
      v4[4] = v18;
    }
LABEL_17:
    --*(_DWORD *)(v2 + 16);
    ++*(_DWORD *)(v2 + 20);
  }
  v11 = &unk_49EE2B0;
  result = v13;
  if ( v13 != 0 && v13 != -8 && v13 != -16 )
    return sub_1649B30(v12);
  return result;
}
