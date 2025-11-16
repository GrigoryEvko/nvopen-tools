// Function: sub_329E210
// Address: 0x329e210
//
__int64 __fastcall sub_329E210(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // eax
  unsigned int v4; // eax
  unsigned int v5; // eax
  __int64 result; // rax

  *(_DWORD *)a1 = *(_DWORD *)a2;
  *(_QWORD *)(a1 + 8) = *(_QWORD *)(a2 + 8);
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
  *(_DWORD *)(a1 + 24) = *(_DWORD *)(a2 + 24);
  v2 = *(_DWORD *)(a2 + 40);
  *(_DWORD *)(a1 + 40) = v2;
  if ( v2 > 0x40 )
    sub_C43780(a1 + 32, (const void **)(a2 + 32));
  else
    *(_QWORD *)(a1 + 32) = *(_QWORD *)(a2 + 32);
  *(_QWORD *)(a1 + 48) = *(_QWORD *)(a2 + 48);
  *(_QWORD *)(a1 + 56) = *(_QWORD *)(a2 + 56);
  *(_QWORD *)(a1 + 64) = *(_QWORD *)(a2 + 64);
  *(_DWORD *)(a1 + 72) = *(_DWORD *)(a2 + 72);
  *(_QWORD *)(a1 + 80) = *(_QWORD *)(a2 + 80);
  *(_QWORD *)(a1 + 88) = *(_QWORD *)(a2 + 88);
  *(_DWORD *)(a1 + 96) = *(_DWORD *)(a2 + 96);
  v3 = *(_DWORD *)(a2 + 112);
  *(_DWORD *)(a1 + 112) = v3;
  if ( v3 > 0x40 )
    sub_C43780(a1 + 104, (const void **)(a2 + 104));
  else
    *(_QWORD *)(a1 + 104) = *(_QWORD *)(a2 + 104);
  *(_QWORD *)(a1 + 120) = *(_QWORD *)(a2 + 120);
  *(_QWORD *)(a1 + 128) = *(_QWORD *)(a2 + 128);
  *(_QWORD *)(a1 + 136) = *(_QWORD *)(a2 + 136);
  *(_DWORD *)(a1 + 144) = *(_DWORD *)(a2 + 144);
  *(_QWORD *)(a1 + 152) = *(_QWORD *)(a2 + 152);
  *(_QWORD *)(a1 + 160) = *(_QWORD *)(a2 + 160);
  *(_DWORD *)(a1 + 168) = *(_DWORD *)(a2 + 168);
  v4 = *(_DWORD *)(a2 + 184);
  *(_DWORD *)(a1 + 184) = v4;
  if ( v4 > 0x40 )
    sub_C43780(a1 + 176, (const void **)(a2 + 176));
  else
    *(_QWORD *)(a1 + 176) = *(_QWORD *)(a2 + 176);
  *(_QWORD *)(a1 + 192) = *(_QWORD *)(a2 + 192);
  *(_QWORD *)(a1 + 200) = *(_QWORD *)(a2 + 200);
  *(_QWORD *)(a1 + 208) = *(_QWORD *)(a2 + 208);
  *(_DWORD *)(a1 + 216) = *(_DWORD *)(a2 + 216);
  *(_QWORD *)(a1 + 224) = *(_QWORD *)(a2 + 224);
  *(_QWORD *)(a1 + 232) = *(_QWORD *)(a2 + 232);
  *(_DWORD *)(a1 + 240) = *(_DWORD *)(a2 + 240);
  v5 = *(_DWORD *)(a2 + 256);
  *(_DWORD *)(a1 + 256) = v5;
  if ( v5 > 0x40 )
    sub_C43780(a1 + 248, (const void **)(a2 + 248));
  else
    *(_QWORD *)(a1 + 248) = *(_QWORD *)(a2 + 248);
  *(_QWORD *)(a1 + 264) = *(_QWORD *)(a2 + 264);
  *(_QWORD *)(a1 + 272) = *(_QWORD *)(a2 + 272);
  result = *(_QWORD *)(a2 + 280);
  *(_QWORD *)(a1 + 280) = result;
  return result;
}
