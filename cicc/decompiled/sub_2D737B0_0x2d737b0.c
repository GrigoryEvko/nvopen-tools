// Function: sub_2D737B0
// Address: 0x2d737b0
//
_QWORD *__fastcall sub_2D737B0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r12
  __int64 v3; // rax
  unsigned int v5; // esi
  int v6; // eax
  _QWORD *v7; // r12
  int v8; // eax
  __int64 v9; // rdx
  unsigned __int64 *v10; // r13
  __int64 v11; // rdx
  _QWORD *v12; // [rsp+0h] [rbp-60h] BYREF
  _QWORD *v13; // [rsp+8h] [rbp-58h] BYREF
  void *v14; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v15[2]; // [rsp+18h] [rbp-48h] BYREF
  __int64 v16; // [rsp+28h] [rbp-38h]
  __int64 v17; // [rsp+30h] [rbp-30h]

  v15[0] = 2;
  v15[1] = 0;
  v16 = a2;
  if ( a2 != 0 && a2 != -4096 && a2 != -8192 )
    sub_BD73F0((__int64)v15);
  v17 = a1;
  v14 = &unk_4A26638;
  if ( !(unsigned __int8)sub_2D69820(a1, (__int64)&v14, &v12) )
  {
    v5 = *(_DWORD *)(a1 + 24);
    v6 = *(_DWORD *)(a1 + 16);
    v7 = v12;
    ++*(_QWORD *)a1;
    v8 = v6 + 1;
    v13 = v7;
    if ( 4 * v8 >= 3 * v5 )
    {
      v5 *= 2;
    }
    else if ( v5 - *(_DWORD *)(a1 + 20) - v8 > v5 >> 3 )
    {
      goto LABEL_12;
    }
    sub_2D72DF0(a1, v5);
    sub_2D69820(a1, (__int64)&v14, &v13);
    v7 = v13;
    v8 = *(_DWORD *)(a1 + 16) + 1;
LABEL_12:
    *(_DWORD *)(a1 + 16) = v8;
    if ( v7[3] == -4096 )
    {
      v9 = v16;
      v3 = -4096;
      v10 = v7 + 1;
      if ( v16 != -4096 )
      {
LABEL_17:
        v7[3] = v9;
        if ( v9 != -4096 && v9 != 0 && v9 != -8192 )
          sub_BD6050(v10, v15[0] & 0xFFFFFFFFFFFFFFF8LL);
        v3 = v16;
      }
    }
    else
    {
      --*(_DWORD *)(a1 + 20);
      v9 = v16;
      v3 = v7[3];
      if ( v16 != v3 )
      {
        v10 = v7 + 1;
        if ( v3 != 0 && v3 != -4096 && v3 != -8192 )
        {
          sub_BD60C0(v7 + 1);
          v9 = v16;
        }
        goto LABEL_17;
      }
    }
    v11 = v17;
    v2 = v7 + 5;
    *v2 = 6;
    v2[1] = 0;
    *(v2 - 1) = v11;
    v2[2] = 0;
    goto LABEL_6;
  }
  v2 = v12 + 5;
  v3 = v16;
LABEL_6:
  v14 = &unk_49DB368;
  if ( v3 != 0 && v3 != -4096 && v3 != -8192 )
    sub_BD60C0(v15);
  return v2;
}
