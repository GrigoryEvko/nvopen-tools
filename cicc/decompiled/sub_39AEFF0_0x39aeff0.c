// Function: sub_39AEFF0
// Address: 0x39aeff0
//
__int64 __fastcall sub_39AEFF0(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  char v4; // r12
  __int64 v5; // rbx
  char v6; // al
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 result; // rax
  bool v10; // cl
  __int64 v11; // rsi
  __int64 v12; // rcx
  __int64 v13; // r13
  const char *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r9
  __int64 v17; // rcx
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rcx
  int v22; // ecx
  char v23; // al
  char v24; // al
  int v25; // [rsp+14h] [rbp-4Ch]
  __int64 v26; // [rsp+18h] [rbp-48h]
  __int64 v27; // [rsp+20h] [rbp-40h]
  __int64 v28; // [rsp+28h] [rbp-38h]

  *(_WORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 26) = 0;
  v3 = *(_QWORD *)(a2 + 416);
  v4 = *(_BYTE *)(a2 + 523);
  v5 = *(_QWORD *)a2;
  v28 = *(_QWORD *)(a2 + 408);
  v6 = sub_396EBA0(*(_QWORD *)(a1 + 8));
  if ( v6 )
    v6 = *(_BYTE *)(a2 + 346);
  *(_BYTE *)(a1 + 26) = v6;
  v8 = sub_396DD80(*(_QWORD *)(a1 + 8));
  result = 0;
  if ( (*(_BYTE *)(v5 + 18) & 8) == 0 )
    goto LABEL_4;
  v26 = v8;
  v25 = *(_DWORD *)(v8 + 12);
  v19 = sub_15E38F0(v5);
  v20 = sub_1649C60(v19);
  if ( *(_BYTE *)(v20 + 16) )
    v20 = 0;
  v27 = v20;
  result = sub_14DD7D0(v20);
  v21 = v27;
  v8 = v26;
  if ( (*(_BYTE *)(v5 + 18) & 8) != 0 && !(_DWORD)result )
  {
    v23 = sub_1560180(v5 + 112, 56);
    v8 = v26;
    if ( v23 || (v24 = sub_1560180(v5 + 112, 30), v8 = v26, !v24) || (*(_BYTE *)(v5 + 18) & 8) != 0 )
    {
      *(_BYTE *)(a1 + 24) = 1;
      result = 0;
      v22 = *(_DWORD *)(v8 + 16);
      goto LABEL_25;
    }
    v21 = v27;
    result = 0;
  }
  if ( (v28 != v3 || v4) && v25 != 255 && v21 )
  {
    *(_BYTE *)(a1 + 24) = 1;
    v22 = *(_DWORD *)(v8 + 16);
LABEL_25:
    v10 = v22 != 255;
    goto LABEL_5;
  }
LABEL_4:
  *(_BYTE *)(a1 + 24) = 0;
  v10 = 0;
LABEL_5:
  v11 = *(_QWORD *)(a1 + 8);
  *(_BYTE *)(a1 + 25) = v10;
  v12 = *(_QWORD *)(v11 + 240);
  if ( *(_DWORD *)(v12 + 348) == 4 )
  {
    v18 = *(unsigned int *)(v12 + 352);
    if ( (_DWORD)v18 )
    {
      if ( (_DWORD)v18 != 6 )
        return (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, __int64, __int64, __int64))(*(_QWORD *)a1 + 72LL))(
                 a1,
                 *(_QWORD *)(a2 + 328),
                 *(_QWORD *)(v11 + 304),
                 v18,
                 v7,
                 v8);
    }
  }
  if ( (_DWORD)result == 7 && v4 != 1 )
  {
    v13 = *(_QWORD *)(a2 + 88);
    v14 = sub_1649960(*(_QWORD *)a2);
    v16 = (__int64)v14;
    v17 = v15;
    if ( v15 )
    {
      if ( *v14 == 1 )
      {
        v17 = v15 - 1;
        v16 = (__int64)(v14 + 1);
      }
    }
    result = sub_39AEF10(a1, v13, v16, v17);
  }
  *(_BYTE *)(a1 + 25) = v4;
  *(_BYTE *)(a1 + 24) = 0;
  return result;
}
