// Function: sub_17D4920
// Address: 0x17d4920
//
__int64 __fastcall sub_17D4920(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // r14
  char v6; // al
  _QWORD *v7; // r13
  __int64 result; // rax
  unsigned int v9; // esi
  int v10; // eax
  int v11; // eax
  __int64 *v12; // rdx
  unsigned __int64 *v13; // r14
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  _QWORD *v19; // [rsp+18h] [rbp-68h] BYREF
  void *v20; // [rsp+20h] [rbp-60h] BYREF
  __int64 v21; // [rsp+28h] [rbp-58h] BYREF
  __int64 v22; // [rsp+30h] [rbp-50h]
  __int64 *v23; // [rsp+38h] [rbp-48h]
  __int64 v24; // [rsp+40h] [rbp-40h]

  v3 = a1 + 304;
  if ( *(_BYTE *)(a1 + 489) )
  {
    v21 = 2;
    v22 = 0;
    v23 = a2;
    if ( !a2 )
      goto LABEL_6;
  }
  else
  {
    v15 = *a2;
    v16 = sub_17CD8D0((_QWORD *)a1, v15);
    a3 = (__int64)v16;
    if ( v16 )
      a3 = sub_15A06D0((__int64 **)v16, v15, v17, v18);
    v21 = 2;
    v22 = 0;
    v23 = a2;
  }
  if ( a2 != (__int64 *)-16LL && a2 != (__int64 *)-8LL )
    sub_164C220((__int64)&v21);
LABEL_6:
  v24 = a1 + 304;
  v20 = &unk_49F04B0;
  v6 = sub_17D3B80(v3, (__int64)&v20, &v19);
  v7 = v19;
  if ( !v6 )
  {
    v9 = *(_DWORD *)(a1 + 328);
    v10 = *(_DWORD *)(a1 + 320);
    ++*(_QWORD *)(a1 + 304);
    v11 = v10 + 1;
    if ( 4 * v11 >= 3 * v9 )
    {
      v9 *= 2;
    }
    else if ( v9 - *(_DWORD *)(a1 + 324) - v11 > v9 >> 3 )
    {
      goto LABEL_14;
    }
    sub_17D3E20(v3, v9);
    sub_17D3B80(v3, (__int64)&v20, &v19);
    v7 = v19;
    v11 = *(_DWORD *)(a1 + 320) + 1;
LABEL_14:
    *(_DWORD *)(a1 + 320) = v11;
    if ( v7[3] == -8 )
    {
      v12 = v23;
      result = -8;
      v13 = v7 + 1;
      if ( v23 != (__int64 *)-8LL )
      {
LABEL_19:
        v7[3] = v12;
        if ( v12 != 0 && v12 + 1 != 0 && v12 != (__int64 *)-16LL )
          sub_1649AC0(v13, v21 & 0xFFFFFFFFFFFFFFF8LL);
        result = (__int64)v23;
      }
    }
    else
    {
      --*(_DWORD *)(a1 + 324);
      v12 = v23;
      result = v7[3];
      if ( v23 != (__int64 *)result )
      {
        v13 = v7 + 1;
        if ( result != 0 && result != -8 && result != -16 )
        {
          sub_1649B30(v7 + 1);
          v12 = v23;
        }
        goto LABEL_19;
      }
    }
    v14 = v24;
    v7[5] = 0;
    v7[4] = v14;
    goto LABEL_8;
  }
  result = (__int64)v23;
LABEL_8:
  v20 = &unk_49EE2B0;
  if ( result != 0 && result != -8 && result != -16 )
    result = sub_1649B30(&v21);
  v7[5] = a3;
  return result;
}
