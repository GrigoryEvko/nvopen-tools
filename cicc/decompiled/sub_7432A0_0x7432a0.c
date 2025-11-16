// Function: sub_7432A0
// Address: 0x7432a0
//
_QWORD *__fastcall sub_7432A0(
        __m128i *a1,
        __m128i *a2,
        __int64 a3,
        _QWORD *a4,
        _DWORD *a5,
        int a6,
        int *a7,
        __int64 *a8)
{
  unsigned __int64 v9; // r12
  int v11; // eax
  __int64 v12; // r13
  _QWORD *v13; // rax
  __int64 v14; // rdi
  int v15; // [rsp+Ch] [rbp-54h]
  int v16; // [rsp+Ch] [rbp-54h]
  _DWORD *v17; // [rsp+10h] [rbp-50h]
  _DWORD *v18; // [rsp+10h] [rbp-50h]
  _QWORD *v19; // [rsp+18h] [rbp-48h]
  _QWORD *v20; // [rsp+18h] [rbp-48h]
  __m128i *v21; // [rsp+20h] [rbp-40h] BYREF
  __int64 v22[7]; // [rsp+28h] [rbp-38h] BYREF

  if ( a1[1].m128i_i8[8] )
  {
    if ( (a1[1].m128i_i8[9] & 1) == 0 )
    {
      v16 = a6;
      v18 = a5;
      v20 = a4;
      v11 = sub_8DBE70(a1->m128i_i64[0]);
      a4 = v20;
      a5 = v18;
      a6 = v16;
      if ( !v11 )
      {
        *a7 = 1;
        return sub_7305B0();
      }
    }
    v15 = a6;
    v17 = a5;
    v19 = a4;
    v21 = (__m128i *)sub_724DC0();
    v9 = sub_7410C0(a1, a2, a3, v19, v17, v15, a7, a8, v21, v22);
    if ( !v9 )
    {
      v12 = v22[0];
      if ( v22[0] )
      {
        v13 = sub_730690(v22[0]);
        v14 = *(_QWORD *)(v12 + 128);
      }
      else
      {
        v13 = sub_73A720(v21, (__int64)a2);
        v14 = v21[8].m128i_i64[0];
      }
      v9 = (unsigned __int64)v13;
      if ( (unsigned int)sub_8D32E0(v14) )
        *(_BYTE *)(v9 + 25) |= 1u;
    }
    sub_724E30((__int64)&v21);
  }
  else
  {
    v9 = (unsigned __int64)sub_7305B0();
  }
  if ( !*a7 )
    return (_QWORD *)v9;
  return sub_7305B0();
}
