// Function: sub_24E84F0
// Address: 0x24e84f0
//
_QWORD *__fastcall sub_24E84F0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  _QWORD *v3; // r12
  __int64 v4; // rax
  unsigned int v6; // esi
  int v7; // eax
  _QWORD *v8; // r12
  int v9; // eax
  __int64 v10; // rdx
  unsigned __int64 *v11; // r13
  __int64 v12; // rdx
  _QWORD *v13; // [rsp+0h] [rbp-60h] BYREF
  _QWORD *v14; // [rsp+8h] [rbp-58h] BYREF
  void *v15; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v16[2]; // [rsp+18h] [rbp-48h] BYREF
  __int64 v17; // [rsp+28h] [rbp-38h]
  __int64 v18; // [rsp+30h] [rbp-30h]

  v2 = *a2;
  v16[0] = 2;
  v16[1] = 0;
  v17 = v2;
  if ( v2 != 0 && v2 != -4096 && v2 != -8192 )
    sub_BD73F0((__int64)v16);
  v18 = a1;
  v15 = &unk_49DD7B0;
  if ( !(unsigned __int8)sub_F9E960(a1, (__int64)&v15, &v13) )
  {
    v6 = *(_DWORD *)(a1 + 24);
    v7 = *(_DWORD *)(a1 + 16);
    v8 = v13;
    ++*(_QWORD *)a1;
    v9 = v7 + 1;
    v14 = v8;
    if ( 4 * v9 >= 3 * v6 )
    {
      v6 *= 2;
    }
    else if ( v6 - *(_DWORD *)(a1 + 20) - v9 > v6 >> 3 )
    {
      goto LABEL_12;
    }
    sub_CF32C0(a1, v6);
    sub_F9E960(a1, (__int64)&v15, &v14);
    v8 = v14;
    v9 = *(_DWORD *)(a1 + 16) + 1;
LABEL_12:
    *(_DWORD *)(a1 + 16) = v9;
    if ( v8[3] == -4096 )
    {
      v10 = v17;
      v4 = -4096;
      v11 = v8 + 1;
      if ( v17 != -4096 )
      {
LABEL_17:
        v8[3] = v10;
        if ( v10 != -4096 && v10 != 0 && v10 != -8192 )
          sub_BD6050(v11, v16[0] & 0xFFFFFFFFFFFFFFF8LL);
        v4 = v17;
      }
    }
    else
    {
      --*(_DWORD *)(a1 + 20);
      v10 = v17;
      v4 = v8[3];
      if ( v17 != v4 )
      {
        v11 = v8 + 1;
        if ( v4 != 0 && v4 != -4096 && v4 != -8192 )
        {
          sub_BD60C0(v8 + 1);
          v10 = v17;
        }
        goto LABEL_17;
      }
    }
    v12 = v18;
    v3 = v8 + 5;
    *v3 = 6;
    v3[1] = 0;
    *(v3 - 1) = v12;
    v3[2] = 0;
    goto LABEL_6;
  }
  v3 = v13 + 5;
  v4 = v17;
LABEL_6:
  v15 = &unk_49DB368;
  if ( v4 != -4096 && v4 != 0 && v4 != -8192 )
    sub_BD60C0(v16);
  return v3;
}
