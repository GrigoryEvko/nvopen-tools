// Function: sub_26A4270
// Address: 0x26a4270
//
__int64 __fastcall sub_26A4270(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  int v4; // eax
  _BYTE *v5; // rdi
  __int64 v6; // r12
  __m128i v7; // rax
  char v8; // al
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  unsigned __int64 v12; // rax
  unsigned __int8 *v13; // rax
  __m128i v14[3]; // [rsp+0h] [rbp-30h] BYREF

  v3 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
  if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
    v3 = *(_QWORD *)(v3 + 24);
  v4 = *(unsigned __int8 *)v3;
  if ( (unsigned __int8)v4 <= 0x1Cu )
  {
    if ( !(_BYTE)v4 )
      goto LABEL_5;
    if ( (_BYTE)v4 == 22 )
    {
      v3 = *(_QWORD *)(v3 + 24);
      goto LABEL_5;
    }
    goto LABEL_22;
  }
  v10 = (unsigned int)(v4 - 34);
  if ( (unsigned __int8)v10 > 0x33u || (v11 = 0x8000000000041LL, !_bittest64(&v11, v10)) )
  {
    v3 = sub_B43CB0(v3);
    goto LABEL_5;
  }
  v12 = sub_250C680((__int64 *)(a1 + 72));
  if ( v12 )
  {
    v3 = *(_QWORD *)(v12 + 24);
    goto LABEL_5;
  }
  v13 = sub_BD3990(*(unsigned __int8 **)(v3 - 32), a2);
  v3 = (__int64)v13;
  if ( !v13 || *v13 )
LABEL_22:
    v3 = 0;
LABEL_5:
  v14[0] = (__m128i)(v3 & 0xFFFFFFFFFFFFFFFCLL | 1);
  nullsub_1518();
  v5 = (_BYTE *)sub_26A2E60(a2, v3 & 0xFFFFFFFFFFFFFFFCLL | 1, 0, a1, 0);
  if ( v5[97] )
  {
    v6 = *(int *)(a1 + 100);
    v7.m128i_i64[0] = (*(__int64 (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v5 + 120LL))(v5, v6);
    v14[0] = v7;
    v8 = *(_BYTE *)(a1 + 16 * v6 + 112);
    if ( v7.m128i_i8[8] == v8 && (!v8 || *(_QWORD *)(a1 + 16 * v6 + 104) == v14[0].m128i_i64[0]) )
    {
      return 1;
    }
    else
    {
      *(__m128i *)(a1 + 16 * v6 + 104) = _mm_loadu_si128(v14);
      return 0;
    }
  }
  else
  {
    *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
    return 0;
  }
}
