// Function: sub_FEECA0
// Address: 0xfeeca0
//
unsigned int __fastcall sub_FEECA0(__int64 a1)
{
  _QWORD *v1; // rax
  __int64 v2; // rax
  __int64 v3; // rdi
  __int128 *v4; // rax

  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = &unk_4F8E808;
  *(_QWORD *)(a1 + 56) = a1 + 104;
  *(_QWORD *)(a1 + 112) = a1 + 160;
  *(_DWORD *)(a1 + 24) = 2;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_49E55B0;
  v1 = (_QWORD *)(a1 + 280);
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 1;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 120) = 1;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_BYTE *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_DWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_DWORD *)(a1 + 88) = 1065353216;
  *(_DWORD *)(a1 + 144) = 1065353216;
  *(_QWORD *)(a1 + 224) = 0;
  *(_DWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  *(_QWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 256) = 0;
  *(_QWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 272) = 1;
  do
  {
    if ( v1 )
      *v1 = -4096;
    v1 += 2;
  }
  while ( (_QWORD *)(a1 + 344) != v1 );
  *(_QWORD *)(a1 + 344) = 0;
  v2 = a1 + 360;
  v3 = a1 + 456;
  *(_QWORD *)(v3 - 104) = 1;
  do
  {
    if ( v2 )
    {
      *(_QWORD *)v2 = -4096;
      *(_DWORD *)(v2 + 8) = 0x7FFFFFFF;
    }
    v2 += 24;
  }
  while ( v3 != v2 );
  v4 = sub_BC2B00();
  return sub_FEEC20((__int64)v4);
}
