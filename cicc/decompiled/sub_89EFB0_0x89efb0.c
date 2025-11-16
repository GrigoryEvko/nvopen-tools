// Function: sub_89EFB0
// Address: 0x89efb0
//
__int64 __fastcall sub_89EFB0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rcx
  __int64 v5; // rcx
  int v6; // eax
  __int64 v7; // rcx
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned int v12; // eax
  __int64 v13; // rax
  __int64 result; // rax

  sub_89EF00(a1, a2);
  *(_DWORD *)(a1 + 100) = 1;
  v3 = qword_4F04C68[0];
  v4 = 776LL * dword_4F04C64;
  *(_DWORD *)(a1 + 156) = dword_4F06650[0];
  v5 = v3 + v4;
  *(_DWORD *)(a1 + 48) = (*(_BYTE *)(v5 + 6) & 8) != 0;
  *(_QWORD *)(a1 + 232) = *(_QWORD *)(v5 + 184);
  v6 = sub_88D5F0();
  ++*(_QWORD *)(a1 + 224);
  v8 = *(_QWORD *)(a1 + 192);
  *(_DWORD *)(a1 + 124) = 1;
  *(_DWORD *)(a1 + 168) = v6 + 1;
  if ( unk_4F04C48 != -1 )
  {
    v9 = v3 + 776LL * unk_4F04C48;
    v10 = *(_QWORD *)(v9 + 360);
    *(_QWORD *)(v8 + 24) = *(_QWORD *)(v9 + 408);
    if ( v10 )
    {
      if ( ((*(_BYTE *)(v10 + 80) - 7) & 0xFD) == 0 )
      {
        v11 = *(_QWORD *)(v10 + 88);
        if ( v11 )
        {
          if ( (*(_BYTE *)(v11 + 170) & 0x10) != 0 && **(_QWORD **)(v11 + 216) )
            *(_QWORD *)(v8 + 96) = v10;
        }
      }
    }
  }
  *(_BYTE *)(v8 + 40) = (*(_BYTE *)(v7 + 9) >> 1) & 7;
  v12 = dword_4F066AC + 1;
  *(_DWORD *)(v8 + 44) = dword_4F066AC + 1;
  dword_4F066AC = v12;
  sub_860400(v8, 0);
  v13 = dword_4F04C64;
  ++*(_QWORD *)(a1 + 216);
  result = qword_4F04C68[0] + 776 * v13;
  *(_BYTE *)(result + 14) |= 4u;
  *(_QWORD *)(result + 616) = a1;
  return result;
}
