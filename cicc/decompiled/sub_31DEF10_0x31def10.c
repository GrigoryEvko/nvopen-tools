// Function: sub_31DEF10
// Address: 0x31def10
//
__int64 __fastcall sub_31DEF10(__int64 a1, __int64 a2, __int64 a3)
{
  bool v3; // r15
  __int64 v8; // r14
  _BYTE *v9; // r15
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // r15
  __int64 result; // rax
  __int64 v14; // rdi
  int v15; // edx
  __int64 v16; // r13
  __int64 v17; // rdi
  char v18; // bl
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 (__fastcall *v22)(__int64, __int64, unsigned __int64); // rbx
  __int64 v23; // rsi
  __int64 v24; // rdi
  unsigned __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 (*v28)(); // rax
  _QWORD v29[8]; // [rsp+0h] [rbp-40h] BYREF

  v3 = 1;
  v8 = sub_31DB510(a1, a3);
  if ( *(_BYTE *)(*(_QWORD *)(a3 + 24) + 8LL) != 13 )
    v3 = *sub_BD3990(*(unsigned __int8 **)(a3 - 32), a3) == 0;
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 200) + 564LL) != 8 )
  {
    if ( (*(_BYTE *)(a3 + 32) & 0xF) != 0 && *(_QWORD *)(*(_QWORD *)(a1 + 208) + 304LL) )
    {
      if ( (((*(_BYTE *)(a3 + 32) & 0xF) + 14) & 0xFu) <= 3 )
      {
        (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 224) + 296LL))(
          *(_QWORD *)(a1 + 224),
          v8,
          26);
        if ( !v3 )
          goto LABEL_10;
        goto LABEL_8;
      }
    }
    else
    {
      (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 224) + 296LL))(*(_QWORD *)(a1 + 224), v8, 9);
    }
    if ( !v3 )
      goto LABEL_10;
LABEL_8:
    (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 224) + 296LL))(*(_QWORD *)(a1 + 224), v8, 2);
    if ( *(_DWORD *)(*(_QWORD *)(a1 + 200) + 564LL) == 1 )
    {
      (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 224) + 312LL))(*(_QWORD *)(a1 + 224), v8);
      (*(void (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 224) + 320LL))(
        *(_QWORD *)(a1 + 224),
        (unsigned int)((*(_BYTE *)(a3 + 32) & 0xFu) - 7 < 2) + 2);
      (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 224) + 328LL))(*(_QWORD *)(a1 + 224), 32);
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 224) + 336LL))(*(_QWORD *)(a1 + 224));
    }
LABEL_10:
    sub_31DE970(a1, v8, (*(_BYTE *)(a3 + 32) >> 4) & 3, 1);
    v9 = (_BYTE *)(*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 240LL))(a1, *(_QWORD *)(a3 - 32));
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 208) + 18LL) && !*v9 )
      (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 224) + 296LL))(
        *(_QWORD *)(a1 + 224),
        v8,
        20);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *))(**(_QWORD **)(a1 + 224) + 272LL))(*(_QWORD *)(a1 + 224), v8, v9);
    v11 = sub_31DE680(a1, a3, v10);
    if ( v8 != v11 )
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *))(**(_QWORD **)(a1 + 224) + 272LL))(
        *(_QWORD *)(a1 + 224),
        v11,
        v9);
    v12 = sub_B325F0(a3);
    result = *(_QWORD *)(a1 + 208);
    if ( *(_BYTE *)(result + 289) )
    {
      if ( (v14 = *(_QWORD *)(a3 + 24), v15 = *(unsigned __int8 *)(v14 + 8), (_BYTE)v15 == 12)
        || (unsigned __int8)v15 <= 3u
        || (_BYTE)v15 == 5
        || (v15 & 0xFD) == 4
        || (v15 & 0xFB) == 0xA
        || ((result = (unsigned int)*(unsigned __int8 *)(v14 + 8) - 15, (unsigned __int8)(*(_BYTE *)(v14 + 8) - 15) <= 3u)
         || v15 == 20)
        && (result = sub_BCEBA0(v14, 0), (_BYTE)result) )
      {
        if ( !v12 || (result = *(_BYTE *)(v12 + 32) & 0xF, (_BYTE)result == 8) )
        {
          v16 = *(_QWORD *)(a3 + 24);
          v17 = a2 + 312;
          v18 = sub_AE5020(a2 + 312, v16);
          v19 = sub_9208B0(v17, v16);
          v29[1] = v20;
          v29[0] = (((unsigned __int64)(v19 + 7) >> 3) + (1LL << v18) - 1) >> v18 << v18;
          v21 = sub_CA1930(v29);
          v22 = *(__int64 (__fastcall **)(__int64, __int64, unsigned __int64))(**(_QWORD **)(a1 + 224) + 448LL);
          v23 = v8;
          v24 = *(_QWORD *)(a1 + 224);
          v25 = sub_E81A90(v21, *(_QWORD **)(a1 + 216), 0, 0);
          return v22(v24, v23, v25);
        }
      }
    }
    return result;
  }
  result = sub_B325F0(a3);
  if ( *(_BYTE *)result != 3 )
  {
    result = (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a1 + 560LL))(a1, a3, v8);
    if ( v3 )
    {
      v22 = *(__int64 (__fastcall **)(__int64, __int64, unsigned __int64))(*(_QWORD *)a1 + 560LL);
      v26 = sub_31DA6B0(a1);
      v25 = 0;
      v27 = v26;
      v28 = *(__int64 (**)())(*(_QWORD *)v26 + 256LL);
      if ( v28 != sub_302E4D0 )
        v25 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v28)(v27, a3, *(_QWORD *)(a1 + 200));
      v23 = a3;
      v24 = a1;
      return v22(v24, v23, v25);
    }
  }
  return result;
}
