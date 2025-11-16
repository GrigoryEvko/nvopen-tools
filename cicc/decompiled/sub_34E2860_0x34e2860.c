// Function: sub_34E2860
// Address: 0x34e2860
//
unsigned __int8 *__fastcall sub_34E2860(
        unsigned int **a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int8 *v8; // r13
  int v10; // eax
  __int64 v11; // rax
  _QWORD *v12; // rdx
  unsigned int v13; // r13d
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned int v17; // ecx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdx
  unsigned __int64 v21; // r15
  __int64 *v22; // rax
  __int64 v23; // r13
  unsigned __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // [rsp+8h] [rbp-A8h]
  __int64 v28; // [rsp+10h] [rbp-A0h]
  __int64 v29; // [rsp+20h] [rbp-90h] BYREF
  __int64 v30; // [rsp+28h] [rbp-88h]
  _QWORD *v31; // [rsp+30h] [rbp-80h] BYREF
  __int64 v32; // [rsp+38h] [rbp-78h]
  _QWORD v33[2]; // [rsp+40h] [rbp-70h] BYREF
  const char *v34; // [rsp+50h] [rbp-60h] BYREF
  __int64 v35; // [rsp+58h] [rbp-58h]
  __int16 v36; // [rsp+70h] [rbp-40h]

  if ( a3 > 0xAE )
  {
    if ( a3 != 248 && a3 != 335 )
    {
      v8 = 0;
      if ( a3 != 237 )
        return v8;
    }
  }
  else
  {
    if ( a3 > 0xAC || a3 == 109 )
    {
LABEL_18:
      v20 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
      v21 = *(_QWORD *)(a2 - 32 * v20);
      v28 = *(_QWORD *)(a2 + 32 * (1 - v20));
      v27 = *(_QWORD *)(a2 + 32 * (2 - v20));
      v34 = *(const char **)(a2 + 8);
      v22 = (__int64 *)sub_B43CA0(a2);
      v23 = sub_B6E160(v22, a3, (__int64)&v34, 1);
      if ( sub_B6AFF0(a3) )
      {
        v34 = sub_BD5D20(a2);
        v36 = 261;
        v32 = v28;
        v35 = v26;
        v31 = (_QWORD *)v21;
        v33[0] = v27;
        v8 = (unsigned __int8 *)sub_B35EA0(a1, v23, (char *)&v31, 3, (__int64)&v34, 0, 0);
      }
      else
      {
        v31 = (_QWORD *)v21;
        v24 = 0;
        v34 = sub_BD5D20(a2);
        v36 = 261;
        v32 = v28;
        v35 = v25;
        v33[0] = v27;
        if ( v23 )
          v24 = *(_QWORD *)(v23 + 24);
        v8 = (unsigned __int8 *)sub_921880(a1, v24, v23, (int)&v31, 3, (__int64)&v34, 0);
      }
      sub_34E2730(v8, a2);
      sub_BD84D0(a2, (__int64)v8);
      sub_B43D60((_QWORD *)a2);
      return v8;
    }
    if ( a3 != 170 )
    {
      v8 = 0;
      if ( a3 != 107 )
        return v8;
      goto LABEL_18;
    }
  }
  v32 = 0x200000000LL;
  v10 = *(_DWORD *)(a2 + 4);
  v31 = v33;
  v11 = v10 & 0x7FFFFFF;
  if ( (_DWORD)v11 != 3 )
  {
    v12 = v33;
    v13 = 0;
    v14 = *(_QWORD *)(a2 - 32 * v11);
    v15 = 0;
    while ( 1 )
    {
      v12[v15] = v14;
      ++v13;
      v16 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
      v17 = v32 + 1;
      LODWORD(v32) = v32 + 1;
      if ( (int)v16 - 3 <= v13 )
        break;
      v14 = *(_QWORD *)(a2 + 32 * (v13 - v16));
      v15 = v17;
      if ( (unsigned __int64)v17 + 1 > HIDWORD(v32) )
      {
        sub_C8D5F0((__int64)&v31, v33, v17 + 1LL, 8u, a5, a6);
        v15 = (unsigned int)v32;
      }
      v12 = v31;
    }
  }
  BYTE4(v30) = 0;
  v34 = sub_BD5D20(a2);
  v18 = *(_QWORD *)(a2 + 8);
  v36 = 261;
  v29 = v18;
  v35 = v19;
  v8 = (unsigned __int8 *)sub_B33D10((__int64)a1, a3, (__int64)&v29, 1, (int)v31, v32, v30, (__int64)&v34);
  sub_34E2730(v8, a2);
  sub_BD84D0(a2, (__int64)v8);
  sub_B43D60((_QWORD *)a2);
  if ( v31 != v33 )
    _libc_free((unsigned __int64)v31);
  return v8;
}
