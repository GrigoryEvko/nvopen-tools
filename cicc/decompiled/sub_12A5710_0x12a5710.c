// Function: sub_12A5710
// Address: 0x12a5710
//
__int64 __fastcall sub_12A5710(_QWORD *a1, __int64 a2, char a3)
{
  __int64 i; // rax
  __int64 v5; // r13
  __int64 v6; // r12
  __int64 *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r10
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 result; // rax
  __int64 v15; // rax
  __int64 *v16; // r14
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rsi
  __int64 v23; // [rsp+18h] [rbp-88h]
  __int64 v24; // [rsp+28h] [rbp-78h] BYREF
  _QWORD v25[2]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v26; // [rsp+40h] [rbp-60h]
  _QWORD v27[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v28; // [rsp+60h] [rbp-40h]

  for ( i = *(_QWORD *)(a2 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v5 = *(_QWORD *)(i + 128);
  v6 = sub_12A2A10((__int64)a1, a2);
  v23 = sub_159C470(*(_QWORD *)(a1[4] + 736LL), v5, 0);
  v7 = (__int64 *)a1[4];
  v26 = 257;
  v8 = *(_QWORD *)v6;
  v9 = v7[94];
  if ( v9 != *(_QWORD *)v6 )
  {
    if ( *(_BYTE *)(v6 + 16) > 0x10u )
    {
      v28 = 257;
      v6 = sub_15FDBD0(47, v6, v9, v27, 0);
      v15 = a1[7];
      if ( v15 )
      {
        v16 = (__int64 *)a1[8];
        sub_157E9D0(v15 + 40, v6);
        v17 = *(_QWORD *)(v6 + 24);
        v18 = *v16;
        *(_QWORD *)(v6 + 32) = v16;
        v18 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v6 + 24) = v18 | v17 & 7;
        *(_QWORD *)(v18 + 8) = v6 + 24;
        *v16 = *v16 & 7 | (v6 + 24);
      }
      sub_164B780(v6, v25);
      v19 = a1[6];
      if ( v19 )
      {
        v24 = a1[6];
        sub_1623A60(&v24, v19, 2);
        v20 = v6 + 48;
        if ( *(_QWORD *)(v6 + 48) )
        {
          sub_161E7C0(v6 + 48);
          v20 = v6 + 48;
        }
        v21 = v24;
        *(_QWORD *)(v6 + 48) = v24;
        if ( v21 )
          sub_1623210(&v24, v21, v20);
      }
      v7 = (__int64 *)a1[4];
      v8 = v7[94];
    }
    else
    {
      v6 = sub_15A46C0(47, v6, v7[94], 0);
      v7 = (__int64 *)a1[4];
      v8 = v7[94];
    }
  }
  v27[0] = v8;
  v10 = *v7;
  if ( a3 )
    v11 = sub_15E26F0(v10, 116, v27, 1);
  else
    v11 = sub_15E26F0(v10, 117, v27, 1);
  v25[1] = v6;
  v28 = 257;
  v25[0] = v23;
  v12 = sub_1285290(a1 + 6, *(_QWORD *)(v11 + 24), v11, (int)v25, 2, (__int64)v27, 0);
  v27[0] = *(_QWORD *)(v12 + 56);
  v13 = sub_16498A0(v12);
  result = sub_1563AB0(v27, v13, 0xFFFFFFFFLL, 30);
  *(_QWORD *)(v12 + 56) = result;
  return result;
}
