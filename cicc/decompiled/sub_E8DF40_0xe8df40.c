// Function: sub_E8DF40
// Address: 0xe8df40
//
__int64 __fastcall sub_E8DF40(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        int a4,
        __int16 a5,
        unsigned __int8 a6,
        char a7,
        int a8,
        int a9,
        __int64 a10)
{
  unsigned int v12; // r12d
  unsigned int v13; // ebx
  __int64 result; // rax
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 v17; // r15
  __int64 v18; // rsi
  __int64 *(__fastcall *v19)(__int64 *, __int64, __int64); // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  _QWORD *v29; // rax
  char v30; // [rsp+4h] [rbp-3Ch]

  v12 = a2;
  v13 = a3;
  result = sub_E99480(a1, a2, a3, a10);
  if ( (_BYTE)result )
  {
    v30 = a6;
    v17 = sub_E6C430(a1[1], a2, v15, a6, v16);
    v18 = v17;
    v19 = *(__int64 *(__fastcall **)(__int64 *, __int64, __int64))(*a1 + 208);
    if ( v19 == sub_E8DC70 )
    {
      sub_E98820(a1, v17, 0);
      sub_E5CB20(a1[37], v17, v20, v21, v22, v23);
      v24 = sub_E8BB10(a1, 0);
      v18 = v17;
      *(_QWORD *)v17 = v24;
      *(_QWORD *)(v17 + 24) = *(_QWORD *)(v24 + 48);
      *(_BYTE *)(v17 + 9) = *(_BYTE *)(v17 + 9) & 0x8F | 0x10;
      sub_E8DAF0((__int64)a1, v17, v25, v26, v27, v28);
    }
    else
    {
      v19(a1, v17, 0);
    }
    v29 = sub_E66210(a1[1], v18);
    return sub_E61510((__int64)v29, a1[1], v17, v12, v13, a4, a5, v30, a7);
  }
  return result;
}
