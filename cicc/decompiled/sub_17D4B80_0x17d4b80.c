// Function: sub_17D4B80
// Address: 0x17d4b80
//
__int64 __fastcall sub_17D4B80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r14
  char v6; // al
  _QWORD *v7; // r13
  unsigned int v8; // esi
  int v9; // eax
  int v10; // eax
  __int64 v11; // rdx
  unsigned __int64 *v12; // r14
  __int64 v13; // rdx
  _QWORD *v14; // [rsp-70h] [rbp-70h] BYREF
  void *v15; // [rsp-68h] [rbp-68h] BYREF
  _QWORD v16[2]; // [rsp-60h] [rbp-60h] BYREF
  __int64 v17; // [rsp-50h] [rbp-50h]
  __int64 v18; // [rsp-48h] [rbp-48h]

  result = *(unsigned int *)(*(_QWORD *)(a1 + 8) + 156LL);
  if ( !(_DWORD)result )
    return result;
  v4 = a1 + 384;
  v16[0] = 2;
  v16[1] = 0;
  v17 = a2;
  if ( a2 != 0 && a2 != -8 && a2 != -16 )
    sub_164C220((__int64)v16);
  v18 = a1 + 384;
  v15 = &unk_49F04B0;
  v6 = sub_17D3B80(v4, (__int64)&v15, &v14);
  v7 = v14;
  if ( !v6 )
  {
    v8 = *(_DWORD *)(a1 + 408);
    v9 = *(_DWORD *)(a1 + 400);
    ++*(_QWORD *)(a1 + 384);
    v10 = v9 + 1;
    if ( 4 * v10 >= 3 * v8 )
    {
      v8 *= 2;
    }
    else if ( v8 - *(_DWORD *)(a1 + 404) - v10 > v8 >> 3 )
    {
      goto LABEL_13;
    }
    sub_17D3E20(v4, v8);
    sub_17D3B80(v4, (__int64)&v15, &v14);
    v7 = v14;
    v10 = *(_DWORD *)(a1 + 400) + 1;
LABEL_13:
    *(_DWORD *)(a1 + 400) = v10;
    if ( v7[3] == -8 )
    {
      v11 = v17;
      result = -8;
      v12 = v7 + 1;
      if ( v17 != -8 )
      {
LABEL_18:
        v7[3] = v11;
        if ( v11 != -8 && v11 != 0 && v11 != -16 )
          sub_1649AC0(v12, v16[0] & 0xFFFFFFFFFFFFFFF8LL);
        result = v17;
      }
    }
    else
    {
      --*(_DWORD *)(a1 + 404);
      v11 = v17;
      result = v7[3];
      if ( v17 != result )
      {
        v12 = v7 + 1;
        if ( result != 0 && result != -8 && result != -16 )
        {
          sub_1649B30(v7 + 1);
          v11 = v17;
        }
        goto LABEL_18;
      }
    }
    v13 = v18;
    v7[5] = 0;
    v7[4] = v13;
    goto LABEL_7;
  }
  result = v17;
LABEL_7:
  v15 = &unk_49EE2B0;
  if ( result != 0 && result != -8 && result != -16 )
    result = sub_1649B30(v16);
  v7[5] = a3;
  return result;
}
