// Function: sub_17D46A0
// Address: 0x17d46a0
//
_QWORD *__fastcall sub_17D46A0(__int64 a1, __int64 a2)
{
  char v2; // al
  _QWORD *v3; // r12
  __int64 v4; // rax
  _QWORD *v5; // r12
  unsigned int v7; // esi
  int v8; // eax
  int v9; // eax
  __int64 v10; // rdx
  unsigned __int64 *v11; // r13
  _QWORD *v12; // [rsp+8h] [rbp-58h] BYREF
  void *v13; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v14[2]; // [rsp+18h] [rbp-48h] BYREF
  __int64 v15; // [rsp+28h] [rbp-38h]
  __int64 v16; // [rsp+30h] [rbp-30h]

  v14[0] = 2;
  v14[1] = 0;
  v15 = a2;
  if ( a2 != 0 && a2 != -8 && a2 != -16 )
    sub_164C220((__int64)v14);
  v16 = a1;
  v13 = &unk_49F04B0;
  v2 = sub_17D3B80(a1, (__int64)&v13, &v12);
  v3 = v12;
  if ( !v2 )
  {
    v7 = *(_DWORD *)(a1 + 24);
    v8 = *(_DWORD *)(a1 + 16);
    ++*(_QWORD *)a1;
    v9 = v8 + 1;
    if ( 4 * v9 >= 3 * v7 )
    {
      v7 *= 2;
    }
    else if ( v7 - *(_DWORD *)(a1 + 20) - v9 > v7 >> 3 )
    {
      goto LABEL_12;
    }
    sub_17D3E20(a1, v7);
    sub_17D3B80(a1, (__int64)&v13, &v12);
    v3 = v12;
    v9 = *(_DWORD *)(a1 + 16) + 1;
LABEL_12:
    *(_DWORD *)(a1 + 16) = v9;
    if ( v3[3] == -8 )
    {
      v10 = v15;
      v4 = -8;
      v11 = v3 + 1;
      if ( v15 != -8 )
      {
LABEL_17:
        v3[3] = v10;
        if ( v10 != -8 && v10 != 0 && v10 != -16 )
          sub_1649AC0(v11, v14[0] & 0xFFFFFFFFFFFFFFF8LL);
        v4 = v15;
      }
    }
    else
    {
      --*(_DWORD *)(a1 + 20);
      v10 = v15;
      v4 = v3[3];
      if ( v15 != v4 )
      {
        v11 = v3 + 1;
        if ( v4 != 0 && v4 != -8 && v4 != -16 )
        {
          sub_1649B30(v3 + 1);
          v10 = v15;
        }
        goto LABEL_17;
      }
    }
    v3[5] = 0;
    v3[4] = v16;
    goto LABEL_6;
  }
  v4 = v15;
LABEL_6:
  v5 = v3 + 5;
  v13 = &unk_49EE2B0;
  if ( v4 != -8 && v4 != 0 && v4 != -16 )
    sub_1649B30(v14);
  return v5;
}
