// Function: sub_3237560
// Address: 0x3237560
//
void __fastcall sub_3237560(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdi
  __int64 *v5; // r13
  __int64 v6; // rax
  _QWORD *v7; // rax
  __int64 v8; // rbx
  _QWORD *v9; // rax
  _BYTE *v10; // rsi
  __int64 v11; // rax
  int v12; // eax
  __m128i v14[4]; // [rsp+10h] [rbp-40h] BYREF

  v4 = a1 + 96;
  v5 = (__int64 *)(a2 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (a2 & 4) != 0 )
  {
    v11 = v5[4];
    v14[0].m128i_i64[0] = v5[3];
    v14[0].m128i_i64[1] = v11;
    v8 = sub_3235E40(v4, v14);
    if ( *(_QWORD *)(v8 + 24) != *(_QWORD *)(v8 + 16) )
      goto LABEL_3;
    *(_QWORD *)v8 = a2;
    v12 = (*(__int64 (__fastcall **)(__int64, __int64))(a1 + 144))(v5[3], v5[4]);
  }
  else
  {
    v6 = *v5;
    v14[0].m128i_i64[0] = (__int64)(v5 + 4);
    v14[0].m128i_i64[1] = v6;
    v7 = (_QWORD *)sub_3235E40(v4, v14);
    v8 = (__int64)v7;
    if ( v7[3] != v7[2] )
      goto LABEL_3;
    *v7 = a2;
    v12 = (*(__int64 (__fastcall **)(__int64 *, __int64))(a1 + 144))(v5 + 4, *v5);
  }
  *(_DWORD *)(v8 + 8) = v12;
LABEL_3:
  v9 = (_QWORD *)sub_A777F0(0x10u, (__int64 *)a1);
  if ( v9 )
  {
    v9[1] = a3;
    *v9 = &unk_4A3D1F0;
  }
  v14[0].m128i_i64[0] = (__int64)v9;
  v10 = *(_BYTE **)(v8 + 24);
  if ( v10 == *(_BYTE **)(v8 + 32) )
  {
    sub_3226B30(v8 + 16, v10, v14);
  }
  else
  {
    if ( v10 )
    {
      *(_QWORD *)v10 = v9;
      v10 = *(_BYTE **)(v8 + 24);
    }
    *(_QWORD *)(v8 + 24) = v10 + 8;
  }
}
