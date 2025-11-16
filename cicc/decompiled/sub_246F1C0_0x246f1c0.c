// Function: sub_246F1C0
// Address: 0x246f1c0
//
__int64 __fastcall sub_246F1C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r14
  _QWORD *v6; // r13
  unsigned int v7; // esi
  int v8; // eax
  _QWORD *v9; // r13
  int v10; // eax
  __int64 v11; // rdx
  unsigned __int64 *v12; // r14
  __int64 v13; // rdx
  _QWORD *v14; // [rsp-78h] [rbp-78h] BYREF
  _QWORD *v15; // [rsp-70h] [rbp-70h] BYREF
  void *v16; // [rsp-68h] [rbp-68h] BYREF
  _QWORD v17[2]; // [rsp-60h] [rbp-60h] BYREF
  __int64 v18; // [rsp-50h] [rbp-50h]
  __int64 v19; // [rsp-48h] [rbp-48h]

  result = *(unsigned int *)(*(_QWORD *)(a1 + 8) + 4LL);
  if ( !(_DWORD)result )
    return result;
  v4 = a1 + 384;
  v17[0] = 2;
  v17[1] = 0;
  v18 = a2;
  if ( a2 != 0 && a2 != -4096 && a2 != -8192 )
    sub_BD73F0((__int64)v17);
  v19 = a1 + 384;
  v16 = &unk_4A16A38;
  if ( !(unsigned __int8)sub_246CEE0(v4, (__int64)&v16, &v14) )
  {
    v7 = *(_DWORD *)(a1 + 408);
    v8 = *(_DWORD *)(a1 + 400);
    v9 = v14;
    ++*(_QWORD *)(a1 + 384);
    v10 = v8 + 1;
    v15 = v9;
    if ( 4 * v10 >= 3 * v7 )
    {
      v7 *= 2;
    }
    else if ( v7 - *(_DWORD *)(a1 + 404) - v10 > v7 >> 3 )
    {
      goto LABEL_13;
    }
    sub_246E1D0(v4, v7);
    sub_246CEE0(v4, (__int64)&v16, &v15);
    v9 = v15;
    v10 = *(_DWORD *)(a1 + 400) + 1;
LABEL_13:
    *(_DWORD *)(a1 + 400) = v10;
    if ( v9[3] == -4096 )
    {
      v11 = v18;
      result = -4096;
      v12 = v9 + 1;
      if ( v18 != -4096 )
      {
LABEL_18:
        v9[3] = v11;
        if ( v11 != -4096 && v11 != 0 && v11 != -8192 )
          sub_BD6050(v12, v17[0] & 0xFFFFFFFFFFFFFFF8LL);
        result = v18;
      }
    }
    else
    {
      --*(_DWORD *)(a1 + 404);
      v11 = v18;
      result = v9[3];
      if ( v18 != result )
      {
        v12 = v9 + 1;
        if ( result != 0 && result != -4096 && result != -8192 )
        {
          sub_BD60C0(v9 + 1);
          v11 = v18;
        }
        goto LABEL_18;
      }
    }
    v13 = v19;
    v9[5] = 0;
    v6 = v9 + 5;
    *(v6 - 1) = v13;
    goto LABEL_7;
  }
  v6 = v14 + 5;
  result = v18;
LABEL_7:
  v16 = &unk_49DB368;
  if ( result != 0 && result != -4096 && result != -8192 )
    result = sub_BD60C0(v17);
  *v6 = a3;
  return result;
}
