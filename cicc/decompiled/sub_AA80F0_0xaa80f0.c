// Function: sub_AA80F0
// Address: 0xaa80f0
//
void __fastcall sub_AA80F0(
        __int64 a1,
        unsigned __int64 *a2,
        char a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8)
{
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // rax
  __int16 v12; // [rsp+10h] [rbp-40h]
  __int64 *v13; // [rsp+18h] [rbp-38h]
  __int64 v14; // [rsp+18h] [rbp-38h]

  if ( a7 == a5 )
  {
    sub_AA7B50(a1, (__int64)a2, a3, a4, (__int64)a7, a6);
  }
  else
  {
    v9 = (__int64)a7;
    if ( *(_BYTE *)(a1 + 40) )
    {
      v12 = a6;
      v14 = a4;
      sub_AA7E90(a1, (__int64)a2, a3, a4, (__int64)a5, a6, (__int64)a7, a8);
      LOWORD(a6) = v12;
      v9 = (__int64)a7;
      a4 = v14;
    }
    if ( (unsigned __int64 *)v9 != a2 )
    {
      v13 = (__int64 *)v9;
      sub_AA4960(a1 + 48, a4 + 48, (__int64)a5, (unsigned __int16)a6, v9);
      if ( v13 != a5 )
      {
        v10 = *v13 & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)((*a5 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v13;
        *v13 = *v13 & 7 | *a5 & 0xFFFFFFFFFFFFFFF8LL;
        v11 = *a2;
        *(_QWORD *)(v10 + 8) = a2;
        v11 &= 0xFFFFFFFFFFFFFFF8LL;
        *a5 = v11 | *a5 & 7;
        *(_QWORD *)(v11 + 8) = a5;
        *a2 = v10 | *a2 & 7;
      }
    }
    sub_AA6320(a1);
  }
}
