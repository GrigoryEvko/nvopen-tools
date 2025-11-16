// Function: sub_8B1A30
// Address: 0x8b1a30
//
void __fastcall sub_8B1A30(__int64 a1, FILE *a2)
{
  char v2; // al
  __int64 v4; // rdi
  __int64 v5; // r12
  __int64 v6; // r13
  _QWORD *v7; // r13
  __int64 v8; // rax

  if ( (*(_BYTE *)(a1 + 207) & 0x20) == 0 )
  {
    v2 = *(_BYTE *)(a1 + 195);
    if ( (v2 & 8) == 0 )
    {
      v4 = *(_QWORD *)(a1 + 344);
      if ( v4 )
      {
        sub_5EB240(v4);
      }
      else if ( (v2 & 1) != 0 && (*(_BYTE *)(a1 + 193) & 0x20) == 0 && !*(_DWORD *)(a1 + 160) )
      {
        v6 = *(_QWORD *)a1;
        sub_8AD0D0(*(_QWORD *)a1, 0, 0);
        v7 = *(_QWORD **)(v6 + 96);
        v8 = v7[2];
        if ( !v8 )
        {
          sub_892270(v7);
          v8 = v7[2];
        }
        if ( (*(_BYTE *)(v8 + 28) & 1) == 0 )
        {
          if ( (unsigned int)sub_899CC0((__int64)v7, 0, 1) )
          {
            ++qword_4D03B78;
            sub_8AB5A0((__int64)v7);
            if ( !--qword_4D03B78 )
              sub_8ACAD0();
          }
        }
      }
      if ( (*(_WORD *)(a1 + 206) & 0x3010) == 0x1000 )
      {
        sub_6854C0(0x9F3u, a2, *(_QWORD *)a1);
        v5 = *(_QWORD *)(a1 + 152);
        *(_QWORD *)(v5 + 160) = sub_72C930();
        *(_BYTE *)(a1 + 207) &= 0xCFu;
      }
    }
  }
}
