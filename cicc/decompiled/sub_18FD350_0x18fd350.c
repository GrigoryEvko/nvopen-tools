// Function: sub_18FD350
// Address: 0x18fd350
//
__int64 __fastcall sub_18FD350(char a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v5; // rax
  __int64 v6; // rax

  if ( a1 )
  {
    v1 = sub_22077B0(160);
    v2 = v1;
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 0;
      *(_QWORD *)(v1 + 16) = &unk_4FAE514;
      *(_QWORD *)(v1 + 80) = v1 + 64;
      *(_QWORD *)(v1 + 88) = v1 + 64;
      *(_QWORD *)(v1 + 128) = v1 + 112;
      *(_QWORD *)(v1 + 136) = v1 + 112;
      *(_DWORD *)(v1 + 24) = 3;
      *(_QWORD *)(v1 + 32) = 0;
      *(_QWORD *)(v1 + 40) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_DWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = 0;
      *(_QWORD *)(v1 + 96) = 0;
      *(_DWORD *)(v1 + 112) = 0;
      *(_QWORD *)(v1 + 120) = 0;
      *(_QWORD *)(v1 + 144) = 0;
      *(_BYTE *)(v1 + 152) = 0;
      *(_QWORD *)v1 = off_49F3258;
      v3 = sub_163A1D0();
      sub_18FD250(v3);
    }
    return v2;
  }
  v5 = sub_22077B0(160);
  v2 = v5;
  if ( !v5 )
    return v2;
  *(_QWORD *)(v5 + 8) = 0;
  *(_QWORD *)(v5 + 16) = &unk_4FAE51C;
  *(_QWORD *)(v5 + 80) = v5 + 64;
  *(_QWORD *)(v5 + 88) = v5 + 64;
  *(_QWORD *)(v5 + 128) = v5 + 112;
  *(_QWORD *)(v5 + 136) = v5 + 112;
  *(_DWORD *)(v5 + 24) = 3;
  *(_QWORD *)(v5 + 32) = 0;
  *(_QWORD *)(v5 + 40) = 0;
  *(_QWORD *)(v5 + 48) = 0;
  *(_DWORD *)(v5 + 64) = 0;
  *(_QWORD *)(v5 + 72) = 0;
  *(_QWORD *)(v5 + 96) = 0;
  *(_DWORD *)(v5 + 112) = 0;
  *(_QWORD *)(v5 + 120) = 0;
  *(_QWORD *)(v5 + 144) = 0;
  *(_BYTE *)(v5 + 152) = 0;
  *(_QWORD *)v5 = off_49F31B0;
  v6 = sub_163A1D0();
  sub_18FD070(v6);
  return v2;
}
