// Function: sub_5C7640
// Address: 0x5c7640
//
__int64 *__fastcall sub_5C7640(__int64 a1, __int64 *a2)
{
  __int64 v3; // r14
  __m128i v4; // xmm2
  __m128i v5; // xmm3
  __m128i v6; // xmm4
  char *v7; // r15
  size_t v8; // rax
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 i; // rax
  _BYTE *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // r14
  int v17; // eax
  __int128 v18; // [rsp+0h] [rbp-90h] BYREF
  __int64 v19; // [rsp+10h] [rbp-80h]
  _OWORD v20[7]; // [rsp+20h] [rbp-70h] BYREF

  v3 = *(_QWORD *)(a1 + 32);
  v4 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v5 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v6 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v20[0] = _mm_loadu_si128(xmmword_4F06660);
  v20[1] = v4;
  v20[2] = v5;
  v20[3] = v6;
  v7 = *(char **)(v3 + 40);
  *((_QWORD *)&v20[0] + 1) = *(_QWORD *)(v3 + 24);
  v8 = strlen(v7);
  sub_878540(v7, v8);
  v9 = sub_7D5DD0(v20, 0);
  v10 = v9;
  if ( v9 && *(_BYTE *)(v9 + 80) == 11 )
  {
    for ( i = *(_QWORD *)(*(_QWORD *)(v9 + 88) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v14 = *(_BYTE **)(i + 168);
    if ( (v14[16] & 2) != 0 )
    {
      if ( *(_QWORD *)v14 && !**(_QWORD **)v14 )
      {
        v15 = a2[15];
        v18 = 0;
        v19 = 0;
        v16 = *(_QWORD *)(*(_QWORD *)v14 + 8LL);
        v17 = sub_72D2E0(v15, 0);
        if ( (unsigned int)sub_8E1010(v17, 0, 0, 0, 0, 0, v16, 0, 0, 1, 1630, (__int64)&v18, 0) )
        {
          if ( DWORD2(v18) )
            sub_684B30(DWORD2(v18), a1 + 56);
        }
        else
        {
          sub_6851C0(1629, a1 + 56);
          *(_BYTE *)(a1 + 8) = 0;
        }
      }
      else
      {
        sub_6851C0(1629, v3 + 24);
        *(_BYTE *)(a1 + 8) = 0;
      }
    }
  }
  else
  {
    sub_684B30(1626, v3 + 24);
    *(_BYTE *)(a1 + 8) = 0;
  }
  if ( *((_BYTE *)a2 + 136) == 3 )
  {
    if ( *((char *)a2 + 169) < 0 )
    {
      sub_684B30(1628, a1 + 56);
      *(_BYTE *)(a1 + 8) = 0;
    }
    else if ( *(_BYTE *)(a1 + 8) )
    {
      sub_8767A0(4, v10, a1 + 56, 1);
      a2[20] = *(_QWORD *)(v10 + 88);
      sub_732AE0();
      *(_BYTE *)(a2[20] + 193) |= 0x40u;
      v11 = *a2;
      *((_BYTE *)a2 + 169) |= 0x10u;
      *(_BYTE *)(v11 + 81) |= 1u;
    }
  }
  else
  {
    sub_684B30(1627, a1 + 56);
    *(_BYTE *)(a1 + 8) = 0;
  }
  return a2;
}
