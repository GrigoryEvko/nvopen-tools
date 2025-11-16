// Function: sub_3749CE0
// Address: 0x3749ce0
//
__int64 __fastcall sub_3749CE0(
        __int64 **a1,
        unsigned int a2,
        unsigned int a3,
        unsigned int a4,
        unsigned __int64 a5,
        unsigned int a6)
{
  unsigned int v6; // r10d
  unsigned int v7; // r13d
  unsigned __int64 v8; // r12
  __int64 v10; // rax
  __int64 (*v11)(); // r11
  __int64 (*v12)(); // rax
  __int64 v13; // rax
  char v14; // cl
  __int64 v15; // rax
  _QWORD *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned int v19; // eax
  __int64 v20; // r9
  __int64 result; // rax
  __int64 v22; // rax
  char v23; // cl
  __int64 v24; // rax
  unsigned __int64 v25; // rax
  unsigned int v26; // eax
  __int64 (*v27)(); // rax
  unsigned int v28; // [rsp+8h] [rbp-48h]
  unsigned int v30; // [rsp+8h] [rbp-48h]
  __int64 v31; // [rsp+10h] [rbp-40h] BYREF
  char v32; // [rsp+18h] [rbp-38h]

  v6 = a6;
  v7 = a3;
  v8 = a5;
  if ( a3 == 58 )
  {
    if ( !a5 || (a5 & (a5 - 1)) != 0 )
      goto LABEL_4;
    _BitScanReverse64(&a5, a5);
    v7 = 190;
    v8 = (int)(63 - (a5 ^ 0x3F));
  }
  else if ( a3 == 60 )
  {
    if ( !a5 || (a5 & (a5 - 1)) != 0 )
      goto LABEL_4;
    _BitScanReverse64(&a5, a5);
    v7 = 192;
    v8 = (int)(63 - (a5 ^ 0x3F));
  }
  else if ( a3 - 190 > 2 )
  {
    goto LABEL_4;
  }
  if ( (unsigned __int16)a2 <= 1u || (unsigned __int16)(a2 - 504) <= 7u )
    goto LABEL_26;
  v22 = 16LL * ((unsigned __int16)a2 - 1);
  v23 = byte_444C4A0[v22 + 8];
  v24 = *(_QWORD *)&byte_444C4A0[v22];
  v32 = v23;
  v31 = v24;
  v25 = sub_CA1930(&v31);
  v6 = a6;
  if ( v25 <= v8 )
    return 0;
LABEL_4:
  v10 = (__int64)*a1;
  v11 = (__int64 (*)())(*a1)[10];
  if ( v11 == sub_3740F20 )
  {
LABEL_5:
    v12 = *(__int64 (**)())(v10 + 88);
    if ( v12 != sub_3740F00 )
    {
      v26 = ((__int64 (__fastcall *)(__int64 **, _QWORD, _QWORD, __int64, unsigned __int64))v12)(a1, v6, v6, 11, v8);
      v20 = v26;
      if ( v26 )
      {
LABEL_24:
        v27 = (__int64 (*)())(*a1)[9];
        if ( v27 != sub_3740EF0 )
          return ((__int64 (__fastcall *)(__int64 **, _QWORD, _QWORD, _QWORD, _QWORD, __int64))v27)(
                   a1,
                   a2,
                   a2,
                   v7,
                   a4,
                   v20);
        return 0;
      }
    }
    if ( (unsigned __int16)a2 > 1u && (unsigned __int16)(a2 - 504) > 7u )
    {
      v13 = 16LL * ((unsigned __int16)a2 - 1);
      v14 = byte_444C4A0[v13 + 8];
      v15 = *(_QWORD *)&byte_444C4A0[v13];
      v32 = v14;
      v31 = v15;
      v28 = sub_CA1930(&v31);
      v16 = (_QWORD *)sub_B2BE50(*a1[5]);
      v17 = sub_BCCE00(v16, v28);
      v18 = sub_ACD640(v17, v8, 0);
      v19 = sub_3746830((__int64 *)a1, v18);
      v20 = v19;
      if ( v19 )
        goto LABEL_24;
      return 0;
    }
LABEL_26:
    BUG();
  }
  v30 = v6;
  result = ((__int64 (__fastcall *)(__int64 **, _QWORD, _QWORD, _QWORD, _QWORD, unsigned __int64))v11)(
             a1,
             a2,
             a2,
             v7,
             a4,
             v8);
  v6 = v30;
  if ( !(_DWORD)result )
  {
    v10 = (__int64)*a1;
    goto LABEL_5;
  }
  return result;
}
