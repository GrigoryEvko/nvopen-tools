// Function: sub_86D170
// Address: 0x86d170
//
__int64 __fastcall sub_86D170(int a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rdi
  int v10; // eax
  __int64 v11; // rsi
  __int64 v12; // rbx
  int v13; // edi
  bool v14; // si
  char v15; // r8
  char v16; // al
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 result; // rax
  __int64 v20; // rax
  char v21; // al
  char v22; // dl
  int v23; // eax
  __int64 v24; // rax
  int v25; // eax
  int v26; // [rsp+Ch] [rbp-34h]
  int v27; // [rsp+Ch] [rbp-34h]

  v9 = qword_4D03B98;
  v10 = unk_4D03B90;
  v11 = 0x2E8BA2E8BA2E8BA3LL * ((qword_4D03B98 - qword_4F5FD90) >> 4) + unk_4D03B90 + 1;
  if ( v11 == qword_4F5FD88 )
  {
    v27 = a4;
    sub_86B3C0(qword_4D03B98, v11, a3, a4, unk_4D03B90, a6);
    v9 = qword_4D03B98;
    v10 = unk_4D03B90;
    LODWORD(a4) = v27;
  }
  unk_4D03B90 = v10 + 1;
  v12 = v9 + 176LL * (v10 + 1);
  *(_DWORD *)v12 = a1;
  v13 = unk_4D03B90;
  *(_WORD *)(v12 + 4) = *(_WORD *)(v12 + 4) & 0xF800 | ((a4 & 1) << 9) | ((a4 & 1) << 10);
  v14 = a1 == 2;
  if ( v13 <= 0 )
  {
    *(_BYTE *)(v12 + 5) = *(_BYTE *)(v12 + 5) & 0xF7 | (8 * v14);
  }
  else
  {
    v15 = *(_BYTE *)(v12 - 171);
    v16 = *(_BYTE *)(v12 + 5);
    if ( (v15 & 4) != 0 )
    {
      v16 |= 4u;
      *(_BYTE *)(v12 + 5) = v16;
    }
    if ( (v15 & 8) != 0 )
      *(_BYTE *)(v12 + 5) = v16 | 8;
    else
      *(_BYTE *)(v12 + 5) = (8 * v14) | v16 & 0xF7;
  }
  *(_BYTE *)(v12 + 5) &= 0xFu;
  *(_QWORD *)(v12 + 8) = a2;
  *(_QWORD *)(v12 + 16) = 0;
  *(_QWORD *)(v12 + 24) = 0;
  *(_QWORD *)(v12 + 32) = 0;
  *(_QWORD *)(v12 + 40) = 0;
  *(_QWORD *)(v12 + 48) = 0;
  *(_QWORD *)(v12 + 56) = 0;
  *(_QWORD *)(v12 + 64) = 0;
  *(_QWORD *)(v12 + 72) = 0;
  *(_QWORD *)(v12 + 80) = 0;
  *(_QWORD *)(v12 + 88) = 0;
  *(_QWORD *)(v12 + 96) = 0;
  *(_QWORD *)(v12 + 128) = a3;
  *(_QWORD *)(v12 + 136) = 0;
  *(_QWORD *)(v12 + 144) = 0;
  *(_DWORD *)(v12 + 152) = -1;
  *(_QWORD *)(v12 + 160) = 0;
  *(_QWORD *)(v12 + 168) = 0;
  if ( a1 )
  {
    if ( v13 > 0 )
    {
      v21 = *(_BYTE *)(v12 + 4);
      v22 = *(_BYTE *)(v12 - 172);
      *(_DWORD *)(v12 + 116) = 0;
      *(_QWORD *)(v12 + 120) = 0;
      *(_BYTE *)(v12 + 4) = v22 & 0x80 | v21 & 0x7F;
      *(_QWORD *)(v12 + 104) = qword_4F5FD78;
      result = (unsigned int)dword_4F5FD80;
      *(_DWORD *)(v12 + 112) = dword_4F5FD80;
      if ( (unsigned int)(a1 - 4) > 3 )
        return result;
    }
    else
    {
      v20 = qword_4F5FD78;
      *(_DWORD *)(v12 + 116) = 0;
      *(_QWORD *)(v12 + 120) = 0;
      *(_QWORD *)(v12 + 104) = v20;
      result = (unsigned int)dword_4F5FD80;
      *(_DWORD *)(v12 + 112) = dword_4F5FD80;
      if ( (unsigned int)(a1 - 4) > 3 )
        return result;
    }
    result = 0x100000001LL;
    dword_4F5FD80 = 0;
    qword_4F5FD78 = 0x100000001LL;
    return result;
  }
  if ( *(_QWORD *)(*(_QWORD *)(a2 + 80) + 16LL) )
  {
    if ( v13 > 0 )
      *(_BYTE *)(v12 + 4) = *(_BYTE *)(v12 - 172) & 0x80 | *(_BYTE *)(v12 + 4) & 0x7F;
  }
  else
  {
    *(_DWORD *)(v12 + 152) = dword_4F04C64;
  }
  v17 = qword_4F5FD78;
  *(_DWORD *)(v12 + 116) = 0;
  *(_QWORD *)(v12 + 120) = 0;
  *(_QWORD *)(v12 + 104) = v17;
  v26 = a4;
  *(_DWORD *)(v12 + 112) = dword_4F5FD80;
  v18 = sub_86B2C0(0);
  if ( dword_4F077C4 == 2 )
  {
    v23 = unk_4D03B90;
    *(_QWORD *)(v18 + 56) = a3;
    if ( v23 > 0 && *(_DWORD *)(v12 + 152) != -1 )
    {
      v24 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 184);
      if ( v24 && *(_BYTE *)(v24 + 28) == 2 && *(_QWORD *)(v24 + 32) )
      {
        *(_WORD *)(v18 + 72) |= 0x120u;
        *(_BYTE *)(v12 + 4) |= 0x20u;
      }
      else
      {
        v25 = *(_DWORD *)(v12 - 176);
        switch ( v25 )
        {
          case 8:
            *(_WORD *)(v18 + 72) |= 0x140u;
            break;
          case 1:
            if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(v12 - 168) + 40LL) - 3) <= 1u )
              *(_BYTE *)(v18 + 73) |= 5u;
            break;
          case 2:
            *(_BYTE *)(v18 + 73) |= 3u;
            break;
        }
      }
    }
  }
  if ( v26 )
    *(_WORD *)(v18 + 72) |= 0x180u;
  return (__int64)sub_86CBE0(v18);
}
