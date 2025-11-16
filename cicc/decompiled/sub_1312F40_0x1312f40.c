// Function: sub_1312F40
// Address: 0x1312f40
//
__int64 __fastcall sub_1312F40(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rax
  int v4; // eax
  unsigned int v5; // r13d
  unsigned int v7; // edx
  __int64 v8; // rdi
  unsigned int v9; // r15d
  unsigned int v10; // r14d
  __int64 v11; // r8
  int v12; // ecx
  __int64 v13; // rsi
  unsigned int v14; // r9d
  unsigned int v15; // ecx
  int v16; // eax
  unsigned int v17; // eax
  __int64 v18; // rax
  __int64 v19; // rcx
  char v20; // cl
  __int64 v21; // rdx
  int v22; // eax

  if ( unk_4C6F210 > 0x1000u )
  {
    if ( unk_4C6F210 > 0x7000000000000000uLL )
    {
      unk_5060A10 = 0;
      v3 = 0;
      goto LABEL_5;
    }
    _BitScanReverse64((unsigned __int64 *)&v19, 2LL * unk_4C6F210 - 1);
    if ( (unsigned __int64)(int)v19 < 7 )
      LOBYTE(v19) = 7;
    v2 = -(1LL << ((unsigned __int8)v19 - 3)) & (unk_4C6F210 + (1LL << ((unsigned __int8)v19 - 3)) - 1);
  }
  else
  {
    v2 = qword_505FA40[byte_5060800[(unsigned __int64)(unk_4C6F210 + 7LL) >> 3]];
  }
  unk_5060A10 = v2;
  if ( v2 <= 0x1000 )
  {
    v3 = (v2 + 7) >> 3;
LABEL_5:
    v4 = byte_5060800[v3] + 1;
    goto LABEL_6;
  }
  if ( v2 > 0x7000000000000000LL )
  {
    v4 = 233;
  }
  else
  {
    v20 = 7;
    _BitScanReverse64((unsigned __int64 *)&v21, 2 * v2 - 1);
    if ( (unsigned int)v21 >= 7 )
      v20 = v21;
    v22 = (((-1LL << (v20 - 3)) & (v2 - 1)) >> (v20 - 3)) & 3;
    if ( (unsigned int)v21 < 6 )
      LODWORD(v21) = 6;
    v4 = v22 + 4 * v21 - 22;
  }
LABEL_6:
  dword_5060A18[0] = v4;
  v5 = sub_130AF40((__int64)&unk_4F96A40);
  if ( (_BYTE)v5 )
    return 1;
  v7 = 36;
  if ( dword_5060A18[0] >= 0x24u )
    v7 = dword_5060A18[0];
  v8 = sub_131C440(a1, a2, 2LL * v7, 64);
  unk_5060A20 = v8;
  if ( !v8 )
  {
    return 1;
  }
  else
  {
    v9 = dword_5060A18[0];
    if ( !dword_5060A18[0] )
      goto LABEL_51;
    v10 = 0;
    do
    {
      if ( v10 <= 0x23 )
      {
        v11 = v10;
        v12 = unk_4C6F204;
        if ( unk_4C6F204 > 0x1FFFu )
          v12 = 0x1FFF;
        v13 = dword_4C6F208[0] - ((unsigned int)((dword_4C6F208[0] & 1) == 0) - 1);
        v14 = 2;
        v15 = ((v12 & 1) == 0) + v12 - 1;
        if ( (unsigned int)v13 < 2 )
          v13 = 2;
        if ( v15 >= 2 )
          v14 = v15;
        if ( (unsigned int)v13 > v14 )
          v13 = v14;
        v16 = *((_DWORD *)&unk_5260DE0 + 10 * v10 + 4) << qword_4C6F1F8;
        if ( (__int64)qword_4C6F1F8 < 0 )
          v16 = *((_DWORD *)&unk_5260DE0 + 10 * v10 + 4) >> -(char)qword_4C6F1F8;
        v17 = v16 - (((v16 & 1) == 0) - 1);
        if ( (unsigned int)v13 < v17 )
        {
          if ( v14 <= v17 )
            v17 = v14;
          v13 = v17;
        }
      }
      else
      {
        v11 = v10;
        v13 = unk_4C6F200;
      }
      ++v10;
      sub_131D380(v8 + 2 * v11, v13);
      v9 = dword_5060A18[0];
      v8 = unk_5060A20;
    }
    while ( dword_5060A18[0] > v10 );
    if ( dword_5060A18[0] <= 0x23u )
    {
LABEL_51:
      while ( 1 )
      {
        v18 = v9++;
        sub_131D380(v8 + 2 * v18, 0);
        if ( v9 == 36 )
          break;
        v8 = unk_5060A20;
      }
      v9 = dword_5060A18[0];
    }
    sub_131D390(unk_5060A20, v9, &qword_4F96AC8, &qword_4F96AC0);
  }
  return v5;
}
