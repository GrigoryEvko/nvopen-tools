// Function: sub_1A6D700
// Address: 0x1a6d700
//
void __fastcall sub_1A6D700(__int64 a1, char a2)
{
  unsigned __int64 v3; // rsi
  _QWORD *v4; // rax
  _DWORD *v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rax
  _DWORD *v9; // r8
  _DWORD *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rax

  *(_QWORD *)(a1 + 16) = &unk_4FB4D74;
  *(_QWORD *)(a1 + 80) = a1 + 64;
  *(_QWORD *)(a1 + 88) = a1 + 64;
  *(_QWORD *)(a1 + 128) = a1 + 112;
  *(_QWORD *)(a1 + 136) = a1 + 112;
  *(_QWORD *)a1 = off_49F57C8;
  *(_QWORD *)(a1 + 232) = a1 + 248;
  *(_QWORD *)(a1 + 320) = a1 + 352;
  *(_QWORD *)(a1 + 328) = a1 + 352;
  *(_BYTE *)(a1 + 153) = a2;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 1;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_BYTE *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 240) = 0x800000000LL;
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 336) = 8;
  *(_DWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 424) = 0;
  *(_QWORD *)(a1 + 432) = 0;
  *(_DWORD *)(a1 + 440) = 0;
  *(_QWORD *)(a1 + 536) = a1 + 552;
  *(_QWORD *)(a1 + 448) = 0;
  *(_QWORD *)(a1 + 456) = 0;
  *(_QWORD *)(a1 + 464) = 0;
  *(_DWORD *)(a1 + 472) = 0;
  *(_QWORD *)(a1 + 480) = 0;
  *(_QWORD *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 496) = 0;
  *(_QWORD *)(a1 + 504) = 0;
  *(_QWORD *)(a1 + 512) = 0;
  *(_QWORD *)(a1 + 520) = 0;
  *(_DWORD *)(a1 + 528) = 0;
  *(_QWORD *)(a1 + 544) = 0x800000000LL;
  *(_QWORD *)(a1 + 616) = 0;
  *(_QWORD *)(a1 + 624) = 0;
  *(_QWORD *)(a1 + 632) = 0;
  *(_DWORD *)(a1 + 640) = 0;
  *(_QWORD *)(a1 + 648) = 0;
  *(_QWORD *)(a1 + 656) = 0;
  *(_QWORD *)(a1 + 664) = 0;
  *(_DWORD *)(a1 + 672) = 0;
  *(_QWORD *)(a1 + 680) = a1 + 696;
  *(_QWORD *)(a1 + 688) = 0x800000000LL;
  v3 = sub_16D5D50();
  v4 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v5 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v6 = v4[2];
        v7 = v4[3];
        if ( v3 <= v4[4] )
          break;
        v4 = (_QWORD *)v4[3];
        if ( !v7 )
          goto LABEL_6;
      }
      v5 = v4;
      v4 = (_QWORD *)v4[2];
    }
    while ( v6 );
LABEL_6:
    if ( v5 != dword_4FA0208 && v3 >= *((_QWORD *)v5 + 4) )
    {
      v8 = *((_QWORD *)v5 + 7);
      v9 = v5 + 12;
      if ( v8 )
      {
        v10 = v5 + 12;
        do
        {
          while ( 1 )
          {
            v11 = *(_QWORD *)(v8 + 16);
            v12 = *(_QWORD *)(v8 + 24);
            if ( *(_DWORD *)(v8 + 32) >= dword_4FB4D88 )
              break;
            v8 = *(_QWORD *)(v8 + 24);
            if ( !v12 )
              goto LABEL_13;
          }
          v10 = (_DWORD *)v8;
          v8 = *(_QWORD *)(v8 + 16);
        }
        while ( v11 );
LABEL_13:
        if ( v9 != v10 && dword_4FB4D88 >= v10[8] && v10[9] )
          *(_BYTE *)(a1 + 153) = byte_4FB4E20;
      }
    }
  }
  v13 = sub_163A1D0();
  sub_1A6D600(v13);
}
