// Function: sub_1E09A30
// Address: 0x1e09a30
//
__int64 __fastcall sub_1E09A30(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rsi
  __int64 result; // rax
  __int64 v8; // rbx
  int v9; // edx
  __int64 v10; // rcx
  __int64 v11; // rdi
  __int64 v12; // rdx
  _QWORD *v13; // r12
  __int64 v14; // rsi
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  bool v20; // al
  bool v21; // zf
  __int64 v22; // [rsp+8h] [rbp-78h] BYREF
  __int64 v23; // [rsp+10h] [rbp-70h]
  __int64 v24; // [rsp+18h] [rbp-68h]
  __int64 v25; // [rsp+20h] [rbp-60h]
  void *v26; // [rsp+30h] [rbp-50h]
  _QWORD v27[2]; // [rsp+38h] [rbp-48h] BYREF
  __int64 v28; // [rsp+48h] [rbp-38h]
  __int64 v29; // [rsp+50h] [rbp-30h]

  v6 = a1[1];
  v23 = 0;
  v22 = v6 & 6;
  v24 = a1[3];
  result = v24;
  if ( v24 != 0 && v24 != -8 && v24 != -16 )
  {
    sub_1649AC0((unsigned __int64 *)&v22, v6 & 0xFFFFFFFFFFFFFFF8LL);
    result = v24;
  }
  v8 = a1[4];
  v25 = v8;
  v9 = *(_DWORD *)(v8 + 24);
  if ( !v9 )
    goto LABEL_5;
  v10 = (unsigned int)(v9 - 1);
  v11 = *(_QWORD *)(v8 + 8);
  v12 = (unsigned int)v10 & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
  v13 = (_QWORD *)(v11 + 48 * v12);
  v14 = v13[3];
  if ( result == v14 )
  {
LABEL_10:
    v15 = v13[5];
    if ( v15 )
      (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64, void *, __int64, __int64))(*(_QWORD *)v15 + 16LL))(
        v15,
        v14,
        v12,
        v10,
        a5,
        a6,
        &unk_49FB768,
        v22,
        v23);
    v28 = -16;
    v26 = &unk_49FB768;
    v29 = 0;
    v16 = v13[3];
    v27[0] = 2;
    v27[1] = 0;
    if ( v16 == -16 )
    {
      v13[4] = 0;
    }
    else
    {
      if ( !v16 || v16 == -8 )
      {
        v13[3] = -16;
        v18 = v28;
        v19 = v29;
        v20 = v28 != 0;
        v21 = v28 == -8;
LABEL_18:
        v13[4] = v19;
        v26 = &unk_49EE2B0;
        if ( v18 != -16 && !v21 && v20 )
          sub_1649B30(v27);
        goto LABEL_20;
      }
      sub_1649B30(v13 + 1);
      v17 = v28;
      v21 = v28 == -8;
      v13[3] = v28;
      if ( v17 != 0 && !v21 && v17 != -16 )
      {
        sub_1649AC0(v13 + 1, v27[0] & 0xFFFFFFFFFFFFFFF8LL);
        v18 = v28;
        v19 = v29;
        v20 = v28 != -8;
        v21 = v28 == 0;
        goto LABEL_18;
      }
      v13[4] = v29;
    }
LABEL_20:
    --*(_DWORD *)(v8 + 16);
    ++*(_DWORD *)(v8 + 20);
    result = v24;
    goto LABEL_5;
  }
  a5 = 1;
  while ( v14 != -8 )
  {
    a6 = (unsigned int)(a5 + 1);
    v12 = (unsigned int)v10 & ((_DWORD)a5 + (_DWORD)v12);
    v13 = (_QWORD *)(v11 + 48LL * (unsigned int)v12);
    v14 = v13[3];
    if ( v14 == result )
      goto LABEL_10;
    a5 = (unsigned int)a6;
  }
LABEL_5:
  if ( result != -8 && result != 0 && result != -16 )
    return sub_1649B30(&v22);
  return result;
}
