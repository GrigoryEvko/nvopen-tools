// Function: sub_390B8A0
// Address: 0x390b8a0
//
void __fastcall sub_390B8A0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  unsigned __int32 v4; // r12d
  unsigned int v5; // r13d
  __m128i v7; // [rsp+10h] [rbp-C0h] BYREF
  __int16 v8; // [rsp+20h] [rbp-B0h]
  __m128i v9; // [rsp+30h] [rbp-A0h] BYREF
  char v10; // [rsp+40h] [rbp-90h]
  char v11; // [rsp+41h] [rbp-8Fh]
  __m128i v12[2]; // [rsp+50h] [rbp-80h] BYREF
  __m128i v13; // [rsp+70h] [rbp-60h] BYREF
  char v14; // [rsp+80h] [rbp-50h]
  char v15; // [rsp+81h] [rbp-4Fh]
  __m128i v16[4]; // [rsp+90h] [rbp-40h] BYREF

  v4 = *(unsigned __int8 *)(a3 + 49);
  if ( *(_BYTE *)(a3 + 49) )
  {
    if ( *(_BYTE *)(a3 + 48) )
    {
      v5 = *(_DWORD *)(a1 + 480);
      if ( v4 + a4 > v5 )
      {
        v4 = v4 + a4 - v5;
        if ( !(*(unsigned __int8 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 120LL))(*(_QWORD *)(a1 + 8)) )
          goto LABEL_8;
        v4 = v5 - a4;
      }
    }
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 8) + 120LL))(
           *(_QWORD *)(a1 + 8),
           a2,
           v4) )
    {
      return;
    }
LABEL_8:
    v7.m128i_i32[0] = v4;
    v13.m128i_i64[0] = (__int64)" bytes";
    v15 = 1;
    v9.m128i_i64[0] = (__int64)"unable to write NOP sequence of ";
    v14 = 3;
    v8 = 265;
    v11 = 1;
    v10 = 3;
    sub_14EC200(v12, &v9, &v7);
    sub_14EC200(v16, v12, &v13);
    sub_16BCFB0((__int64)v16, 1u);
  }
}
