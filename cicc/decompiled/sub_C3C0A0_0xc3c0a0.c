// Function: sub_C3C0A0
// Address: 0xc3c0a0
//
__int64 __fastcall sub_C3C0A0(__int64 *a1, __int64 *a2)
{
  char *v2; // rax
  char v3; // al
  char v4; // bl
  unsigned int v5; // r8d
  bool v6; // bl
  const __m128i *v8; // rax
  __m128i v9; // xmm0
  __int64 v10; // rdx
  int v11; // eax
  unsigned int v12; // r8d
  unsigned __int8 v13; // al
  __int64 v14; // rdx
  char v15; // al
  int v16; // eax
  unsigned int v17; // [rsp+10h] [rbp-F0h]
  unsigned int v18; // [rsp+10h] [rbp-F0h]
  unsigned int v19; // [rsp+10h] [rbp-F0h]
  bool v20; // [rsp+2Fh] [rbp-D1h] BYREF
  __int64 v21[4]; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v22[2]; // [rsp+50h] [rbp-B0h] BYREF
  char v23; // [rsp+64h] [rbp-9Ch]
  _QWORD v24[4]; // [rsp+70h] [rbp-90h] BYREF
  _QWORD v25[4]; // [rsp+90h] [rbp-70h] BYREF
  __m128i v26; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v27; // [rsp+C0h] [rbp-40h]
  int v28; // [rsp+C8h] [rbp-38h]

  v2 = (char *)sub_C94E20(qword_4F863F0);
  if ( v2 )
    v3 = *v2;
  else
    v3 = qword_4F863F0[2];
  if ( v3 && (_DWORD *)*a1 == dword_3F657C0 )
  {
    sub_C36070((__int64)a1, 0, 0, 0);
    return 1;
  }
  else
  {
    v4 = *((_BYTE *)a1 + 20);
    v5 = sub_C395E0(a1, a2);
    v6 = (v4 & 8) != 0;
    if ( v5 == 2 )
    {
      sub_C33EB0(v21, a2);
      if ( !(unsigned int)sub_C3ADF0((__int64)v21, (__int64)a2, 1) )
        sub_C3BE30(a1, v21);
      sub_C33EB0(v22, a2);
      v8 = (const __m128i *)*a1;
      *((_BYTE *)a1 + 20) &= ~8u;
      v23 &= ~8u;
      v9 = _mm_loadu_si128(v8);
      v26 = v9;
      v10 = v8[1].m128i_i64[0];
      v26.m128i_i32[0] = v9.m128i_i32[0] + 1;
      v27 = v10;
      LODWORD(v8) = v8[1].m128i_i32[2];
      v26.m128i_i32[1] = v9.m128i_i32[1] - 1;
      v26.m128i_i32[2] = v9.m128i_i32[2] + 2;
      v28 = (int)v8;
      sub_C33EB0(v24, a1);
      sub_C396A0((__int64)v24, &v26, 1, &v20);
      sub_C33EB0(v25, v22);
      sub_C396A0((__int64)v25, &v26, 1, &v20);
      v17 = sub_C3ADF0((__int64)v24, (__int64)v24, 1);
      v11 = sub_C37950((__int64)v24, (__int64)v25);
      v12 = v17;
      if ( v11 == 2 )
      {
        sub_C3B1F0((__int64)a1, (__int64)v22, 1);
        sub_C3B1F0((__int64)v24, (__int64)v25, 1);
        v19 = sub_C3B1F0((__int64)v24, (__int64)v25, 1);
        v16 = sub_C37950((__int64)v24, (__int64)v25);
        v12 = v19;
        if ( (unsigned int)(v16 - 1) <= 1 )
          v12 = sub_C3B1F0((__int64)a1, (__int64)v22, 1);
      }
      v13 = *((_BYTE *)a1 + 20);
      if ( (v13 & 7) == 3 )
      {
        v14 = *a1;
        v15 = (8 * v6) | v13 & 0xF7;
        *((_BYTE *)a1 + 20) = v15;
        if ( *(_DWORD *)(v14 + 20) == 2 )
          *((_BYTE *)a1 + 20) = v15 & 0xF7;
      }
      else
      {
        *((_BYTE *)a1 + 20) = v13 & 0xF7 | (8 * (((v13 >> 3) ^ v6) & 1));
      }
      v18 = v12;
      sub_C338F0((__int64)v25);
      sub_C338F0((__int64)v24);
      sub_C338F0((__int64)v22);
      sub_C338F0((__int64)v21);
      return v18;
    }
  }
  return v5;
}
