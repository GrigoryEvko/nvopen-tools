// Function: sub_1A40A10
// Address: 0x1a40a10
//
void __fastcall sub_1A40A10(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  int v3; // edx
  char v4; // al
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  int v10; // r12d

  if ( (unsigned int)sub_16AF4C0(&dword_4FB404C, 1, 0) )
  {
    do
    {
      v10 = dword_4FB404C;
      sub_16AF4B0();
    }
    while ( v10 != 2 );
  }
  else
  {
    v1 = sub_22077B0(200);
    v2 = v1;
    if ( v1 )
    {
      *(_QWORD *)v1 = &unk_49EED30;
      v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
      *(_WORD *)(v1 + 12) &= 0xF000u;
      *(_QWORD *)(v1 + 72) = qword_4FA01C0;
      *(_QWORD *)(v1 + 88) = v1 + 120;
      *(_QWORD *)(v1 + 96) = v1 + 120;
      *(_DWORD *)(v1 + 8) = v3;
      *(_WORD *)(v1 + 176) = 256;
      *(_QWORD *)(v1 + 168) = &unk_49E74E8;
      *(_QWORD *)(v1 + 16) = 0;
      *(_QWORD *)(v1 + 24) = 0;
      *(_QWORD *)v1 = &unk_49EEC70;
      *(_QWORD *)(v1 + 32) = 0;
      *(_QWORD *)(v1 + 40) = 0;
      *(_QWORD *)(v1 + 184) = &unk_49EEDB0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 80) = 0;
      *(_QWORD *)(v1 + 104) = 4;
      *(_DWORD *)(v1 + 112) = 0;
      *(_BYTE *)(v1 + 152) = 0;
      *(_BYTE *)(v1 + 160) = 0;
      sub_16B8280(v1, "scalarize-load-store", 0x14u);
      *(_QWORD *)(v2 + 40) = "Allow the scalarizer pass to scalarize loads and store";
      v4 = *(_BYTE *)(v2 + 12);
      *(_QWORD *)(v2 + 48) = 54;
      *(_BYTE *)(v2 + 160) = 0;
      *(_WORD *)(v2 + 176) = 256;
      *(_BYTE *)(v2 + 12) = v4 & 0x9F | 0x20;
      sub_16B88A0(v2);
    }
    v5 = sub_3946700();
    sub_3946900(v5, &unk_4FB4048, v2);
    v6 = sub_22077B0(80);
    if ( v6 )
    {
      *(_QWORD *)(v6 + 8) = 27;
      *(_QWORD *)v6 = "Scalarize vector operations";
      *(_QWORD *)(v6 + 16) = "scalarizer";
      *(_QWORD *)(v6 + 32) = &unk_4FB4050;
      *(_WORD *)(v6 + 40) = 0;
      *(_QWORD *)(v6 + 24) = 10;
      *(_BYTE *)(v6 + 42) = 0;
      *(_QWORD *)(v6 + 48) = 0;
      *(_QWORD *)(v6 + 56) = 0;
      *(_QWORD *)(v6 + 64) = 0;
      *(_QWORD *)(v6 + 72) = sub_1A40C60;
    }
    sub_163A800(a1, (_QWORD *)v6, 1, v7, v8, v9);
    sub_16AF4B0();
    dword_4FB404C = 2;
  }
}
