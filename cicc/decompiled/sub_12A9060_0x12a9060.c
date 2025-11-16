// Function: sub_12A9060
// Address: 0x12a9060
//
_QWORD *__fastcall sub_12A9060(_QWORD *a1, _BYTE *a2, _BYTE *a3)
{
  __int64 v4; // r12
  _QWORD *result; // rax
  __int64 v6; // r12
  _QWORD *v7; // r13
  __int64 v8; // rax
  _QWORD *v9; // r14
  __int64 v10; // rdi
  unsigned __int64 *v11; // r13
  __int64 v12; // rax
  unsigned __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // r14
  bool v18; // cc
  __int64 v19; // rax
  __int64 v20; // rdi
  unsigned __int64 *v21; // r14
  __int64 v22; // rax
  unsigned __int64 v23; // rsi
  __int64 v24; // rsi
  __int64 v25; // rsi
  __int64 v26; // [rsp+18h] [rbp-D8h]
  __int64 v27; // [rsp+20h] [rbp-D0h]
  __int64 v28; // [rsp+28h] [rbp-C8h]
  __int64 v31; // [rsp+48h] [rbp-A8h]
  __int64 v32; // [rsp+58h] [rbp-98h] BYREF
  char v33[16]; // [rsp+60h] [rbp-90h] BYREF
  __int16 v34; // [rsp+70h] [rbp-80h]
  char v35[16]; // [rsp+80h] [rbp-70h] BYREF
  __int16 v36; // [rsp+90h] [rbp-60h]
  _BYTE v37[16]; // [rsp+A0h] [rbp-50h] BYREF
  __int16 v38; // [rsp+B0h] [rbp-40h]

  v4 = *(_QWORD *)a2;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 16 )
    return sub_12A8F50(a1 + 6, (__int64)a2, (__int64)a3, 0);
  v28 = sub_1643350(a1[5]);
  v27 = **(_QWORD **)(v4 + 16);
  result = *(_QWORD **)(v4 + 32);
  if ( (_DWORD)result )
  {
    v6 = 0;
    v26 = (unsigned int)result;
    do
    {
      v16 = sub_15A0680(v28, v6, 0);
      v36 = 257;
      v17 = v16;
      v31 = sub_12A8800(a1 + 6, v27, a3, v6, (__int64)v35);
      v18 = a2[16] <= 0x10u;
      v34 = 257;
      if ( v18 && *(_BYTE *)(v17 + 16) <= 0x10u )
      {
        v7 = (_QWORD *)sub_15A37D0(a2, v17, 0);
      }
      else
      {
        v38 = 257;
        v19 = sub_1648A60(56, 2);
        v7 = (_QWORD *)v19;
        if ( v19 )
          sub_15FA320(v19, a2, v17, v37, 0);
        v20 = a1[7];
        if ( v20 )
        {
          v21 = (unsigned __int64 *)a1[8];
          sub_157E9D0(v20 + 40, v7);
          v22 = v7[3];
          v23 = *v21;
          v7[4] = v21;
          v23 &= 0xFFFFFFFFFFFFFFF8LL;
          v7[3] = v23 | v22 & 7;
          *(_QWORD *)(v23 + 8) = v7 + 3;
          *v21 = *v21 & 7 | (unsigned __int64)(v7 + 3);
        }
        sub_164B780(v7, v33);
        v24 = a1[6];
        if ( v24 )
        {
          v32 = a1[6];
          sub_1623A60(&v32, v24, 2);
          if ( v7[6] )
            sub_161E7C0(v7 + 6);
          v25 = v32;
          v7[6] = v32;
          if ( v25 )
            sub_1623210(&v32, v25, v7 + 6);
        }
      }
      v38 = 257;
      v8 = sub_1648A60(64, 2);
      v9 = (_QWORD *)v8;
      if ( v8 )
        sub_15F9650(v8, v7, v31, 0, 0);
      v10 = a1[7];
      if ( v10 )
      {
        v11 = (unsigned __int64 *)a1[8];
        sub_157E9D0(v10 + 40, v9);
        v12 = v9[3];
        v13 = *v11;
        v9[4] = v11;
        v13 &= 0xFFFFFFFFFFFFFFF8LL;
        v9[3] = v13 | v12 & 7;
        *(_QWORD *)(v13 + 8) = v9 + 3;
        *v11 = *v11 & 7 | (unsigned __int64)(v9 + 3);
      }
      result = (_QWORD *)sub_164B780(v9, v37);
      v14 = a1[6];
      if ( v14 )
      {
        v32 = a1[6];
        result = (_QWORD *)sub_1623A60(&v32, v14, 2);
        if ( v9[6] )
          result = (_QWORD *)sub_161E7C0(v9 + 6);
        v15 = v32;
        v9[6] = v32;
        if ( v15 )
          result = (_QWORD *)sub_1623210(&v32, v15, v9 + 6);
      }
      ++v6;
    }
    while ( v26 != v6 );
  }
  return result;
}
