// Function: sub_A78DB0
// Address: 0xa78db0
//
__int64 __fastcall sub_A78DB0(_QWORD *a1, int a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  __int64 v6; // r12
  _QWORD *v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rbx
  __int64 v10; // rdi
  _BYTE *v11; // rsi
  __int64 result; // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // r14
  unsigned int v16; // edx
  __int64 v17; // rdx
  unsigned int v18; // edx
  unsigned int v19; // edx
  __int64 v20; // rsi
  int v21; // [rsp+8h] [rbp-F8h]
  __int64 v23; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v24; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v25; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v26; // [rsp+18h] [rbp-E8h]
  _QWORD *v27; // [rsp+28h] [rbp-D8h]
  __int64 v28; // [rsp+28h] [rbp-D8h]
  __int64 v29; // [rsp+30h] [rbp-D0h] BYREF
  unsigned __int64 v30; // [rsp+38h] [rbp-C8h] BYREF
  _QWORD v31[2]; // [rsp+40h] [rbp-C0h] BYREF
  int v32; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v33; // [rsp+54h] [rbp-ACh]

  v4 = a3;
  v6 = a3;
  v7 = (_QWORD *)*a1;
  v21 = a4;
  v33 = a4;
  v8 = 32 * a4;
  v27 = v7;
  v9 = a3 + v8;
  v31[0] = &v32;
  v32 = a2;
  v31[1] = 0x2000000003LL;
  v23 = v8;
  if ( a3 != a3 + v8 )
  {
    do
    {
      sub_C439F0(v4, v31);
      v10 = v4 + 16;
      v4 += 32;
      sub_C439F0(v10, v31);
    }
    while ( v9 != v4 );
  }
  v11 = v31;
  result = sub_C65B40(v27 + 50, v31, &v29, off_49D9AB0);
  if ( !result )
  {
    v13 = v27[330];
    v27[340] += v23 + 24;
    v14 = (v13 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v27[331] >= v23 + 24 + v14 && v13 )
      v27[330] = v23 + 24 + v14;
    else
      v14 = sub_9D1E70((__int64)(v27 + 330), v23 + 24, v23 + 24, 3);
    *(_QWORD *)v14 = 0;
    v15 = v14 + 24;
    *(_BYTE *)(v14 + 8) = 5;
    *(_DWORD *)(v14 + 12) = a2;
    *(_DWORD *)(v14 + 16) = v21;
    if ( a3 == v9 )
    {
LABEL_17:
      v26 = v14;
      sub_C657C0(v27 + 50, v14, v29, off_49D9AB0);
      result = v26;
      v30 = v26;
      v11 = (_BYTE *)v27[212];
      if ( v11 == (_BYTE *)v27[213] )
      {
        sub_A78C20((__int64)(v27 + 211), v11, &v30);
        result = v26;
      }
      else
      {
        if ( v11 )
        {
          *(_QWORD *)v11 = v26;
          v11 = (_BYTE *)v27[212];
        }
        v11 += 8;
        v27[212] = v11;
      }
      goto LABEL_4;
    }
    while ( 1 )
    {
      v18 = *(_DWORD *)(v6 + 8);
      *(_DWORD *)(v15 + 8) = v18;
      if ( v18 <= 0x40 )
      {
        *(_QWORD *)v15 = *(_QWORD *)v6;
        v16 = *(_DWORD *)(v6 + 24);
        *(_DWORD *)(v15 + 24) = v16;
        if ( v16 <= 0x40 )
          goto LABEL_13;
LABEL_16:
        v20 = v6 + 16;
        v6 += 32;
        v25 = v14;
        sub_C43780(v15 + 16, v20);
        v15 += 32LL;
        v14 = v25;
        if ( v9 == v6 )
          goto LABEL_17;
      }
      else
      {
        v24 = v14;
        sub_C43780(v15, v6);
        v19 = *(_DWORD *)(v6 + 24);
        v14 = v24;
        *(_DWORD *)(v15 + 24) = v19;
        if ( v19 > 0x40 )
          goto LABEL_16;
LABEL_13:
        v17 = *(_QWORD *)(v6 + 16);
        v6 += 32;
        v15 += 32LL;
        *(_QWORD *)(v15 - 16) = v17;
        if ( v9 == v6 )
          goto LABEL_17;
      }
    }
  }
LABEL_4:
  if ( (int *)v31[0] != &v32 )
  {
    v28 = result;
    _libc_free(v31[0], v11);
    return v28;
  }
  return result;
}
