// Function: sub_E5CBA0
// Address: 0xe5cba0
//
void __fastcall sub_E5CBA0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v4; // r13
  __int64 v5; // r15
  unsigned int v6; // r14d
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __m128i v14[2]; // [rsp+10h] [rbp-120h] BYREF
  __int16 v15; // [rsp+30h] [rbp-100h]
  __m128i v16[2]; // [rsp+40h] [rbp-F0h] BYREF
  char v17; // [rsp+60h] [rbp-D0h]
  char v18; // [rsp+61h] [rbp-CFh]
  __m128i v19[3]; // [rsp+70h] [rbp-C0h] BYREF
  __m128i v20[2]; // [rsp+A0h] [rbp-90h] BYREF
  char v21; // [rsp+C0h] [rbp-70h]
  char v22; // [rsp+C1h] [rbp-6Fh]
  __m128i v23[6]; // [rsp+D0h] [rbp-60h] BYREF

  LODWORD(v4) = *(unsigned __int8 *)(a3 + 30);
  if ( *(_BYTE *)(a3 + 30) )
  {
    v5 = *(_QWORD *)(a3 + 32);
    if ( (*(_BYTE *)(a3 + 29) & 2) != 0 )
    {
      v6 = *(_DWORD *)(a1 + 368);
      if ( (int)v4 + a4 > v6 )
      {
        v4 = (_DWORD)v4 + a4 - v6;
        if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a1 + 8) + 192LL))(
                *(_QWORD *)(a1 + 8),
                a2,
                v4,
                v5) )
          goto LABEL_8;
        LODWORD(v4) = v6 - a4;
      }
    }
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64, _QWORD, __int64))(**(_QWORD **)(a1 + 8) + 192LL))(
           *(_QWORD *)(a1 + 8),
           a2,
           (unsigned int)v4,
           v5) )
    {
      return;
    }
LABEL_8:
    v22 = 1;
    v14[0].m128i_i32[0] = v4;
    v20[0].m128i_i64[0] = (__int64)" bytes";
    v16[0].m128i_i64[0] = (__int64)"unable to write NOP sequence of ";
    v21 = 3;
    v15 = 265;
    v18 = 1;
    v17 = 3;
    sub_9C6370(v19, v16, v14, v8, v9, v10);
    sub_9C6370(v23, v19, v20, v11, v12, v13);
    sub_C64D30((__int64)v23, 1u);
  }
}
