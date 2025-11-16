// Function: sub_1AEBFA0
// Address: 0x1aebfa0
//
__int64 __fastcall sub_1AEBFA0(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  unsigned int v9; // r13d
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  unsigned __int64 v13; // r15
  __int64 v14; // rax
  unsigned __int64 v15; // rbx
  __int64 v16; // rdi
  __int64 v17; // rax
  double v18; // xmm4_8
  double v19; // xmm5_8
  int v20; // edx
  unsigned __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v24; // rax

  v9 = 0;
  v11 = sub_157EBA0(a1);
  v12 = *(_QWORD *)(a1 + 48);
  v13 = v11;
  while ( 1 )
  {
    v14 = v12 - 24;
    if ( !v12 )
      v14 = 0;
    if ( v13 == v14 )
      return v9;
    v15 = *(_QWORD *)(v13 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v15 )
      BUG();
    if ( *(_QWORD *)(v15 - 16) )
    {
      v16 = *(_QWORD *)(v15 - 24);
      if ( *(_BYTE *)(v16 + 8) != 10 )
      {
        v17 = sub_1599EF0((__int64 **)v16);
        sub_164D160(v15 - 24, v17, a2, a3, a4, a5, v18, v19, a8, a9);
        goto LABEL_9;
      }
      v13 = v15 - 24;
    }
    else
    {
LABEL_9:
      v20 = *(unsigned __int8 *)(v15 - 8);
      v21 = (unsigned int)(v20 - 34);
      if ( (unsigned int)v21 <= 0x36 && (v22 = 0x40018000000001LL, _bittest64(&v22, v21))
        || *(_BYTE *)(*(_QWORD *)(v15 - 24) + 8LL) == 10 )
      {
        v12 = *(_QWORD *)(a1 + 48);
        v13 = v15 - 24;
      }
      else
      {
        if ( (_BYTE)v20 != 78
          || (v24 = *(_QWORD *)(v15 - 48), *(_BYTE *)(v24 + 16))
          || (*(_BYTE *)(v24 + 33) & 0x20) == 0
          || (unsigned int)(*(_DWORD *)(v24 + 36) - 35) > 3 )
        {
          ++v9;
        }
        sub_15F20C0((_QWORD *)(v15 - 24));
        v12 = *(_QWORD *)(a1 + 48);
      }
    }
  }
}
