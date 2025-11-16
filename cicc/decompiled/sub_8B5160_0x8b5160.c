// Function: sub_8B5160
// Address: 0x8b5160
//
__int64 __fastcall sub_8B5160(__m128i *a1, __int64 a2, __int64 a3, const __m128i *a4)
{
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // rbx
  __m128i *v8; // rax
  unsigned int v9; // r12d
  __int64 v11; // [rsp+0h] [rbp-40h]
  __int64 *v12[7]; // [rsp+8h] [rbp-38h] BYREF

  v12[0] = (__int64 *)sub_72F240(a4);
  if ( qword_4F04C18
    && (v5 = qword_4F04C18[2]) != 0
    && (v6 = *(_QWORD *)(v5 + 8)) != 0
    && (v7 = *(_QWORD *)(v6 + 88)) != 0 )
  {
    v11 = *(_QWORD *)(v6 + 80);
    v8 = sub_72F240(*(const __m128i **)(v6 + 88));
    *(_QWORD *)(v6 + 88) = v8;
    *(_QWORD *)(v6 + 80) = v8->m128i_i64[0];
    v9 = sub_8B3500(a1, a2, (__int64 *)v12, a3, 0);
    if ( v12[0] )
      sub_725130(v12[0]);
    sub_725130(*(__int64 **)(v6 + 88));
    *(_QWORD *)(v6 + 88) = v7;
    *(_QWORD *)(v6 + 80) = v11;
  }
  else
  {
    v9 = sub_8B3500(a1, a2, (__int64 *)v12, a3, 0);
    if ( v12[0] )
      sub_725130(v12[0]);
  }
  return v9;
}
