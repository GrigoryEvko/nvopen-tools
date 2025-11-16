// Function: sub_2ABA9E0
// Address: 0x2aba9e0
//
__int64 __fastcall sub_2ABA9E0(__int64 a1, char a2, __int64 a3, unsigned __int8 *a4, __int64 a5)
{
  __int64 v7; // rdx
  __int64 v8; // rax
  unsigned __int64 v9; // rax
  __int16 v10; // ax
  __int64 result; // rax
  char v12; // al
  char v13; // dl
  __int64 v14; // rdx
  unsigned __int8 v15; // al
  unsigned __int8 v16; // al
  int v17; // eax
  __int64 v18; // [rsp+10h] [rbp-40h] BYREF
  __int64 v19[7]; // [rsp+18h] [rbp-38h] BYREF

  v18 = *((_QWORD *)a4 + 6);
  if ( v18 )
  {
    sub_2AAAFA0(&v18);
    v19[0] = v18;
    if ( v18 )
      sub_2AAAFA0(v19);
  }
  else
  {
    v19[0] = 0;
  }
  *(_BYTE *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_4A231A8;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 64) = a3;
  *(_QWORD *)(a1 + 40) = &unk_4A23170;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 56) = 0x200000001LL;
  v7 = *(unsigned int *)(a3 + 24);
  if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 28) )
  {
    sub_C8D5F0(a3 + 16, (const void *)(a3 + 32), v7 + 1, 8u, a5, v7 + 1);
    v7 = *(unsigned int *)(a3 + 24);
  }
  *(_QWORD *)(*(_QWORD *)(a3 + 16) + 8 * v7) = a1 + 40;
  ++*(_DWORD *)(a3 + 24);
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 40) = &unk_4A23AA8;
  v8 = v19[0];
  *(_QWORD *)a1 = &unk_4A23A70;
  *(_QWORD *)(a1 + 88) = v8;
  if ( v8 )
    sub_2AAAFA0((__int64 *)(a1 + 88));
  sub_9C6650(v19);
  sub_2BF0340(a1 + 96, 1, a4, a1);
  *(_QWORD *)a1 = &unk_4A231C8;
  *(_QWORD *)(a1 + 40) = &unk_4A23200;
  *(_QWORD *)(a1 + 96) = &unk_4A23238;
  sub_9C6650(&v18);
  *(_QWORD *)a1 = &unk_4A23258;
  *(_QWORD *)(a1 + 96) = &unk_4A232C8;
  v9 = *a4;
  *(_QWORD *)(a1 + 40) = &unk_4A23290;
  if ( (unsigned __int8)(v9 - 82) <= 1u )
  {
    v10 = *((_WORD *)a4 + 1);
    *(_BYTE *)(a1 + 152) = 0;
    result = v10 & 0x3F;
    *(_DWORD *)(a1 + 156) = result;
    return result;
  }
  if ( (_BYTE)v9 == 58 )
  {
    *(_BYTE *)(a1 + 152) = 2;
LABEL_13:
    v12 = a4[1] >> 1;
LABEL_14:
    v13 = v12 & 1;
    result = v12 & 1 | *(_BYTE *)(a1 + 156) & 0xFEu;
    *(_BYTE *)(a1 + 156) = v13 | *(_BYTE *)(a1 + 156) & 0xFE;
    return result;
  }
  if ( (unsigned __int8)v9 <= 0x36u )
  {
    v14 = 0x40540000000000LL;
    if ( _bittest64(&v14, v9) )
    {
      v15 = a4[1];
      *(_BYTE *)(a1 + 152) = 1;
      result = *(_BYTE *)(a1 + 156) & 0xFC | (v15 >> 1) & 3u;
      *(_BYTE *)(a1 + 156) = result;
      return result;
    }
    if ( (unsigned int)(unsigned __int8)v9 - 48 <= 1 )
      goto LABEL_17;
  }
  else
  {
    if ( (unsigned __int8)(v9 - 55) <= 1u )
    {
LABEL_17:
      *(_BYTE *)(a1 + 152) = 3;
      goto LABEL_13;
    }
    if ( (_BYTE)v9 == 63 )
    {
      *(_BYTE *)(a1 + 152) = 4;
      result = sub_B4DE20((__int64)a4);
      *(_DWORD *)(a1 + 156) = result;
      return result;
    }
  }
  if ( (((_BYTE)v9 - 68) & 0xFB) == 0 )
  {
    *(_BYTE *)(a1 + 152) = 6;
    v12 = sub_B44910((__int64)a4);
    goto LABEL_14;
  }
  result = sub_920620((__int64)a4);
  if ( (_BYTE)result )
  {
    v16 = a4[1];
    *(_BYTE *)(a1 + 152) = 5;
    v17 = v16 >> 1;
    if ( v17 == 127 )
      v17 = -1;
    LODWORD(v19[0]) = v17;
    sub_2C1AC80(&v18, v19);
    result = (unsigned __int8)v18;
    *(_BYTE *)(a1 + 156) = v18;
  }
  else
  {
    *(_BYTE *)(a1 + 152) = 7;
    *(_DWORD *)(a1 + 156) = 0;
  }
  return result;
}
