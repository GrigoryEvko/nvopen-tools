// Function: sub_140AD80
// Address: 0x140ad80
//
__m128i *__fastcall sub_140AD80(__m128i *a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // rax
  __int64 v5; // rbx
  __m128i v6; // xmm0
  __int32 v8; // ecx
  __int32 v9; // edx
  char v10; // [rsp+7h] [rbp-49h] BYREF
  __int64 v11; // [rsp+8h] [rbp-48h] BYREF
  __m128i v12; // [rsp+10h] [rbp-40h] BYREF
  char v13; // [rsp+20h] [rbp-30h]

  v4 = sub_140ABA0(a2, 0, &v10);
  if ( !v4 )
    goto LABEL_9;
  v5 = v4;
  if ( !v10 )
  {
    sub_140A980(&v12, v4, 0x1Fu, a3);
    if ( v13 )
    {
      v6 = _mm_loadu_si128(&v12);
      a1[1].m128i_i8[0] = 1;
      *a1 = v6;
      return a1;
    }
  }
  v11 = sub_1560310(v5 + 112, 0xFFFFFFFFLL, 2);
  if ( v11 )
  {
    sub_155D750(&v12, &v11);
    v8 = v12.m128i_i32[0];
    v9 = -1;
    if ( v12.m128i_i8[8] )
      v9 = v12.m128i_i32[1];
    a1->m128i_i32[1] = *(_DWORD *)(v5 + 20) & 0xFFFFFFF;
    a1[1].m128i_i8[0] = 1;
    a1->m128i_i8[0] = 3;
    a1->m128i_i32[2] = v8;
    a1->m128i_i32[3] = v9;
    return a1;
  }
  else
  {
LABEL_9:
    a1[1].m128i_i8[0] = 0;
    return a1;
  }
}
