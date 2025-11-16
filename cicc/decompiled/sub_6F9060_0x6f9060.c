// Function: sub_6F9060
// Address: 0x6f9060
//
void __fastcall sub_6F9060(__int64 a1)
{
  __int64 *v1; // r13
  __int64 v2; // rax
  char i; // dl
  char v4; // bl
  unsigned int v5; // r14d
  __int64 j; // rdx
  unsigned int v7; // edx
  __int64 v8; // rax
  __int64 ***v9; // r15
  __int64 *v10; // rax
  __int64 v11; // r15
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 **v22; // [rsp+8h] [rbp-198h]
  __m128i v23[4]; // [rsp+10h] [rbp-190h] BYREF
  __int64 v24; // [rsp+54h] [rbp-14Ch]
  __int64 v25; // [rsp+5Ch] [rbp-144h]

  v1 = *(__int64 **)(a1 + 144);
  if ( *((_BYTE *)v1 + 24) )
  {
    v2 = *v1;
    for ( i = *(_BYTE *)(*v1 + 140); i == 12; i = *(_BYTE *)(v2 + 140) )
      v2 = *(_QWORD *)(v2 + 160);
    if ( i )
    {
      v4 = *((_BYTE *)v1 + 25);
      v5 = sub_85E8D0();
      for ( j = *(_QWORD *)(qword_4F04C68[0] + 776LL * (int)v5 + 216) >> 3; ; LODWORD(j) = v7 + 1 )
      {
        v7 = qword_4F04C10[1] & j;
        v8 = *qword_4F04C10 + 16LL * v7;
        v9 = *(__int64 ****)v8;
        if ( *(_QWORD *)(qword_4F04C68[0] + 776LL * (int)v5 + 216) == *(_QWORD *)v8 )
          break;
        if ( !v9 )
          goto LABEL_11;
      }
      v9 = *(__int64 ****)(v8 + 8);
LABEL_11:
      *(_BYTE *)(a1 + 20) &= ~0x10u;
      v10 = v1;
      if ( *((_BYTE *)v1 + 24) == 3 )
      {
LABEL_16:
        v22 = sub_5EA830(v9, (__int64 *)v10[7], 0);
        v11 = sub_5F6E90((__int64)v9, (__int64)v22);
        v12 = sub_830AC0(v22, v5, 1);
        *(_QWORD *)(v12 + 28) = *(__int64 *)((char *)v1 + 28);
        *(_QWORD *)(v12 + 36) = *(__int64 *)((char *)v1 + 28);
        *(_QWORD *)(v12 + 44) = *(__int64 *)((char *)v1 + 44);
        if ( (unsigned int)sub_8D32E0(*(_QWORD *)(v11 + 120)) )
        {
          sub_6E7150((__int64 *)v12, (__int64)v23);
          v24 = *(_QWORD *)(a1 + 68);
          v25 = *(_QWORD *)(a1 + 76);
          sub_6F82C0((__int64)v23, (__int64)v23, v14, v15, v16, v17);
          v12 = sub_6F6F40(v23, 0, v18, v19, v20, v21);
        }
        *(__m128i *)v1 = _mm_loadu_si128((const __m128i *)v12);
        *((__m128i *)v1 + 1) = _mm_loadu_si128((const __m128i *)(v12 + 16));
        *((__m128i *)v1 + 2) = _mm_loadu_si128((const __m128i *)(v12 + 32));
        *((__m128i *)v1 + 3) = _mm_loadu_si128((const __m128i *)(v12 + 48));
        *((__m128i *)v1 + 4) = _mm_loadu_si128((const __m128i *)(v12 + 64));
        v1[10] = *(_QWORD *)(v12 + 80);
        v13 = *v1;
        *((_BYTE *)v1 + 25) = *((_BYTE *)v1 + 25) & 0xFC | v4 & 3;
        *(_QWORD *)a1 = v13;
      }
      else
      {
        while ( *((_BYTE *)v10 + 56) != 95 )
        {
          v10 = (__int64 *)v10[9];
          if ( *((_BYTE *)v10 + 24) == 3 )
            goto LABEL_16;
        }
      }
    }
  }
}
