// Function: sub_32376B0
// Address: 0x32376b0
//
void __fastcall sub_32376B0(__int64 a1, __int64 a2, __int64 a3, unsigned int *a4, unsigned __int8 *a5)
{
  __int64 *v7; // rdx
  __int64 v9; // rdi
  __int64 v10; // rax
  _QWORD *v11; // rax
  _QWORD *v12; // rbx
  unsigned int v13; // r15d
  __int64 v14; // rax
  __int64 v15; // r12
  _BYTE *v16; // rsi
  __int64 v17; // rax
  _QWORD *v18; // rax
  int v19; // eax
  _QWORD *v20; // [rsp+10h] [rbp-50h]
  unsigned int v22; // [rsp+18h] [rbp-48h]
  __m128i v23[4]; // [rsp+20h] [rbp-40h] BYREF

  v7 = (__int64 *)(a2 & 0xFFFFFFFFFFFFFFF8LL);
  v9 = a1 + 96;
  if ( (a2 & 4) != 0 )
  {
    v17 = v7[4];
    v23[0].m128i_i64[0] = v7[3];
    v23[0].m128i_i64[1] = v17;
    v18 = (_QWORD *)sub_3235E40(v9, v23);
    v12 = v18;
    if ( v18[3] != v18[2] )
      goto LABEL_3;
    *v18 = a2;
    v19 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(a1 + 144))(
            *(_QWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24),
            *(_QWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 32));
  }
  else
  {
    v10 = *v7;
    v23[0].m128i_i64[0] = (__int64)(v7 + 4);
    v20 = v7 + 4;
    v23[0].m128i_i64[1] = v10;
    v11 = (_QWORD *)sub_3235E40(v9, v23);
    v12 = v11;
    if ( v11[3] != v11[2] )
      goto LABEL_3;
    *v11 = a2;
    v19 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD))(a1 + 144))(v20, *(_QWORD *)(a2 & 0xFFFFFFFFFFFFFFF8LL));
  }
  *((_DWORD *)v12 + 2) = v19;
LABEL_3:
  v13 = *a5;
  v22 = *a4;
  v14 = sub_A777F0(0x30u, (__int64 *)a1);
  v15 = v14;
  if ( v14 )
    sub_37236A0(v14, a3, v22, v13);
  v23[0].m128i_i64[0] = v15;
  v16 = (_BYTE *)v12[3];
  if ( v16 == (_BYTE *)v12[4] )
  {
    sub_3226B30((__int64)(v12 + 2), v16, v23);
  }
  else
  {
    if ( v16 )
    {
      *(_QWORD *)v16 = v15;
      v16 = (_BYTE *)v12[3];
    }
    v12[3] = v16 + 8;
  }
}
