// Function: sub_1A48EA0
// Address: 0x1a48ea0
//
__int64 __fastcall sub_1A48EA0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  const void *v6; // r8
  __int64 v7; // r14
  __int64 v8; // r15
  __int64 v9; // r13
  __int64 v10; // rdx
  __int64 v11; // r12
  __int64 *v12; // r13
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // r15
  __int64 v16; // rdx
  __int64 result; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // [rsp+0h] [rbp-70h]
  __int64 v22; // [rsp+8h] [rbp-68h]
  __int64 v23; // [rsp+8h] [rbp-68h]
  const void *v24; // [rsp+8h] [rbp-68h]
  const char *v25; // [rsp+10h] [rbp-60h] BYREF
  __int64 v26; // [rsp+18h] [rbp-58h]
  _QWORD v27[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v28; // [rsp+30h] [rbp-40h]

  v6 = (const void *)(a1 + 96);
  v7 = 8LL * a2;
  v8 = a2 - 1;
  v9 = 8 * v8;
  v10 = *(_QWORD *)a1;
  v11 = *(_QWORD *)(*(_QWORD *)a1 + v7);
  if ( a2 )
  {
    while ( (unsigned __int8)(*(_BYTE *)(v11 + 16) - 60) <= 0xCu )
    {
      v18 = *(unsigned int *)(a1 + 88);
      if ( (unsigned int)v18 >= *(_DWORD *)(a1 + 92) )
      {
        v24 = v6;
        sub_16CD150(a1 + 80, v6, 0, 8, (int)v6, a6);
        v18 = *(unsigned int *)(a1 + 88);
        v6 = v24;
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 80) + 8 * v18) = v11;
      v19 = *(_QWORD *)a1;
      ++*(_DWORD *)(a1 + 88);
      *(_QWORD *)(v19 + v7) = 0;
      v10 = *(_QWORD *)a1;
      v7 = v9;
      v11 = *(_QWORD *)(*(_QWORD *)a1 + v9);
      if ( !(_DWORD)v8 )
        goto LABEL_10;
      v9 -= 8;
      LODWORD(v8) = v8 - 1;
    }
    v21 = *(_QWORD *)(v11 - 48);
    v22 = *(_QWORD *)(v10 + 8LL * (unsigned int)v8);
    v12 = (__int64 *)sub_1A48D70(a1, *(_QWORD *)(v11 + 24LL * (v22 == v21) - 48));
    v13 = sub_1A48EA0(a1, (unsigned int)v8);
    v14 = v22;
    v15 = (__int64 *)v13;
    v23 = *(_QWORD *)(a1 + 224);
    if ( v14 == v21 )
    {
      v25 = sub_1649960(v11);
      v26 = v20;
      v28 = 261;
      v27[0] = &v25;
      result = sub_15FB440((unsigned int)*(unsigned __int8 *)(v11 + 16) - 24, v15, (__int64)v12, (__int64)v27, v23);
    }
    else
    {
      v25 = sub_1649960(v11);
      v28 = 261;
      v26 = v16;
      v27[0] = &v25;
      result = sub_15FB440((unsigned int)*(unsigned __int8 *)(v11 + 16) - 24, v12, (__int64)v15, (__int64)v27, v23);
    }
    *(_QWORD *)(*(_QWORD *)a1 + v7) = result;
  }
  else
  {
LABEL_10:
    result = sub_1A48D70(a1, v11);
    *(_QWORD *)(*(_QWORD *)a1 + v7) = result;
  }
  return result;
}
