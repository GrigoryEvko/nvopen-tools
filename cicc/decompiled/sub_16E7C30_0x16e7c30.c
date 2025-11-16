// Function: sub_16E7C30
// Address: 0x16e7c30
//
void __fastcall sub_16E7C30(int *a1, __int64 a2)
{
  int v2; // edx
  int v3; // eax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __m128i *v6; // rax
  __int64 v7; // rcx
  _QWORD v8[2]; // [rsp+0h] [rbp-50h] BYREF
  __m128i v9; // [rsp+10h] [rbp-40h] BYREF
  _BYTE v10[40]; // [rsp+20h] [rbp-30h] BYREF

  *(_QWORD *)a1 = &unk_49EFB70;
  if ( a1[9] < 0 )
    goto LABEL_5;
  if ( *((_QWORD *)a1 + 1) != *((_QWORD *)a1 + 3) )
    sub_16E7BA0((__int64 *)a1);
  if ( *((_BYTE *)a1 + 40) && (v3 = sub_16C6980(a1[9], a2), v5 = v4, (v2 = v3) != 0) )
  {
    a1[12] = v3;
    *((_QWORD *)a1 + 7) = v5;
  }
  else
  {
LABEL_5:
    v2 = a1[12];
  }
  if ( v2 )
  {
    (*(void (__fastcall **)(_BYTE *))(**((_QWORD **)a1 + 7) + 32LL))(v10);
    v6 = (__m128i *)sub_2241130(v10, 0, 0, "IO failure on output stream: ", 29);
    v8[0] = &v9;
    if ( (__m128i *)v6->m128i_i64[0] == &v6[1] )
    {
      v9 = _mm_loadu_si128(v6 + 1);
    }
    else
    {
      v8[0] = v6->m128i_i64[0];
      v9.m128i_i64[0] = v6[1].m128i_i64[0];
    }
    v7 = v6->m128i_i64[1];
    v6[1].m128i_i8[0] = 0;
    v8[1] = v7;
    v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
    v6->m128i_i64[1] = 0;
    sub_16BD160((__int64)v8, 0);
  }
  *(_QWORD *)a1 = &unk_49EFD28;
  sub_16E7960((__int64)a1);
}
