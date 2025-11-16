// Function: sub_246EF60
// Address: 0x246ef60
//
__int64 __fastcall sub_246EF60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 *v6; // r12
  __int64 result; // rax
  unsigned int v8; // esi
  int v9; // eax
  _QWORD *v10; // r12
  int v11; // eax
  __int64 v12; // rdx
  unsigned __int64 *v13; // r14
  __int64 v14; // rdx
  __int64 v15; // rsi
  _QWORD *v16; // rax
  _QWORD *v17; // [rsp+0h] [rbp-70h] BYREF
  _QWORD *v18; // [rsp+8h] [rbp-68h] BYREF
  void *v19; // [rsp+10h] [rbp-60h] BYREF
  __int64 v20; // [rsp+18h] [rbp-58h] BYREF
  __int64 v21; // [rsp+20h] [rbp-50h]
  __int64 v22; // [rsp+28h] [rbp-48h]
  __int64 v23; // [rsp+30h] [rbp-40h]

  v3 = a1 + 304;
  if ( *(_BYTE *)(a1 + 633) )
  {
    v20 = 2;
    v21 = 0;
    v22 = a2;
    if ( !a2 )
      goto LABEL_6;
  }
  else
  {
    v15 = *(_QWORD *)(a2 + 8);
    v16 = sub_2463540((__int64 *)a1, v15);
    a3 = (__int64)v16;
    if ( v16 )
      a3 = sub_AD6530((__int64)v16, v15);
    v20 = 2;
    v21 = 0;
    v22 = a2;
  }
  if ( a2 != -8192 && a2 != -4096 )
    sub_BD73F0((__int64)&v20);
LABEL_6:
  v23 = a1 + 304;
  v19 = &unk_4A16A38;
  if ( !(unsigned __int8)sub_246CEE0(v3, (__int64)&v19, &v17) )
  {
    v8 = *(_DWORD *)(a1 + 328);
    v9 = *(_DWORD *)(a1 + 320);
    v10 = v17;
    ++*(_QWORD *)(a1 + 304);
    v11 = v9 + 1;
    v18 = v10;
    if ( 4 * v11 >= 3 * v8 )
    {
      v8 *= 2;
    }
    else if ( v8 - *(_DWORD *)(a1 + 324) - v11 > v8 >> 3 )
    {
      goto LABEL_14;
    }
    sub_246E1D0(v3, v8);
    sub_246CEE0(v3, (__int64)&v19, &v18);
    v10 = v18;
    v11 = *(_DWORD *)(a1 + 320) + 1;
LABEL_14:
    *(_DWORD *)(a1 + 320) = v11;
    if ( v10[3] == -4096 )
    {
      v12 = v22;
      result = -4096;
      v13 = v10 + 1;
      if ( v22 != -4096 )
      {
LABEL_19:
        v10[3] = v12;
        if ( v12 != 0 && v12 != -4096 && v12 != -8192 )
          sub_BD6050(v13, v20 & 0xFFFFFFFFFFFFFFF8LL);
        result = v22;
      }
    }
    else
    {
      --*(_DWORD *)(a1 + 324);
      v12 = v22;
      result = v10[3];
      if ( v22 != result )
      {
        v13 = v10 + 1;
        if ( result != 0 && result != -4096 && result != -8192 )
        {
          sub_BD60C0(v10 + 1);
          v12 = v22;
        }
        goto LABEL_19;
      }
    }
    v14 = v23;
    v6 = v10 + 5;
    *v6 = 0;
    *(v6 - 1) = v14;
    goto LABEL_8;
  }
  v6 = v17 + 5;
  result = v22;
LABEL_8:
  v19 = &unk_49DB368;
  if ( result != 0 && result != -4096 && result != -8192 )
    result = sub_BD60C0(&v20);
  *v6 = a3;
  return result;
}
