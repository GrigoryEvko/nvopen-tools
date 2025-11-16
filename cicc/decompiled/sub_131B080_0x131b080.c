// Function: sub_131B080
// Address: 0x131b080
//
__int64 __fastcall sub_131B080(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v5; // rax
  __int64 v8; // r12
  unsigned __int64 v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rbx
  __int64 v13; // r12
  __int64 result; // rax
  char v15; // cl
  unsigned __int64 v16; // rax
  int v17; // eax

  v5 = *(_QWORD *)(a2 + 16);
  v8 = a4;
  if ( v5 )
  {
    v10 = v5 + 1;
    if ( v5 + 1 > 0x1000 )
    {
      a4 = 0x7000000000000000LL;
      if ( v10 > 0x7000000000000000LL )
      {
        v11 = 3864;
      }
      else
      {
        v15 = 7;
        _BitScanReverse64(&v10, 2 * v10 - 1);
        if ( (unsigned int)v10 >= 7 )
          v15 = v10;
        v16 = ((-1LL << (v15 - 3)) & v5) >> (v15 - 3);
        a4 = 6;
        v17 = v16 & 3;
        if ( (unsigned int)v10 < 6 )
          v10 = 6;
        v11 = 16LL * (unsigned int)(v17 + 4 * v10 - 14) + 8;
      }
    }
    else
    {
      v10 = (unsigned __int64)byte_5060800;
      v11 = 16LL * ((unsigned int)byte_5060800[(v5 + 8) >> 3] - 1) + 168;
    }
    sub_133F890(a1 + v11, a2, v10, a4);
  }
  *(_QWORD *)(a1 + 3880) += a5;
  v12 = v8 + a5;
  v13 = v8 - a3;
  *(_QWORD *)(a1 + 3888) += ((v12 + 4095) & 0xFFFFFFFFFFFFF000LL) - ((v13 + 4095) & 0xFFFFFFFFFFFFF000LL);
  result = dword_4F96B94;
  if ( dword_4F96B94 && !unk_505F9C8 && (dword_4F96B94 == 2 || *(_BYTE *)(a1 + 144)) )
    *(_QWORD *)(a1 + 3904) += (((v12 + 0x1FFFFF) & 0xFFFFFFFFFFE00000LL) - ((v13 + 0x1FFFFF) & 0xFFFFFFFFFFE00000LL)) >> 21;
  return result;
}
