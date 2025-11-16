// Function: sub_391D7F0
// Address: 0x391d7f0
//
__int64 *__fastcall sub_391D7F0(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v5; // rax

  v3 = *a2;
  *a2 = 0;
  v5 = sub_22077B0(0x3D8u);
  if ( v5 )
  {
    *(_QWORD *)(v5 + 8) = a3;
    *(_QWORD *)v5 = off_4A3EC58;
    *(_DWORD *)(v5 + 16) = 1;
    *(_QWORD *)(v5 + 24) = v3;
    *(_QWORD *)(v5 + 32) = 0;
    *(_QWORD *)(v5 + 40) = 0;
    *(_QWORD *)(v5 + 48) = 0;
    *(_QWORD *)(v5 + 64) = 0;
    *(_QWORD *)(v5 + 72) = 0;
    *(_QWORD *)(v5 + 80) = 0;
    *(_QWORD *)(v5 + 96) = 0;
    *(_QWORD *)(v5 + 104) = 0;
    *(_QWORD *)(v5 + 112) = 0;
    *(_DWORD *)(v5 + 120) = 0;
    *(_QWORD *)(v5 + 128) = 0;
    *(_QWORD *)(v5 + 136) = 0;
    *(_QWORD *)(v5 + 144) = 0;
    *(_DWORD *)(v5 + 152) = 0;
    *(_QWORD *)(v5 + 160) = 0;
    *(_QWORD *)(v5 + 168) = 0;
    *(_QWORD *)(v5 + 176) = 0;
    *(_DWORD *)(v5 + 184) = 0;
    *(_QWORD *)(v5 + 192) = 0;
    *(_QWORD *)(v5 + 200) = 0;
    *(_QWORD *)(v5 + 208) = 0;
    *(_DWORD *)(v5 + 216) = 0;
    *(_QWORD *)(v5 + 224) = 0;
    *(_QWORD *)(v5 + 232) = 0;
    *(_QWORD *)(v5 + 240) = 0;
    *(_QWORD *)(v5 + 248) = 0;
    *(_QWORD *)(v5 + 256) = 0;
    *(_QWORD *)(v5 + 264) = 0;
    *(_DWORD *)(v5 + 272) = 0;
    *(_QWORD *)(v5 + 344) = v5 + 360;
    *(_QWORD *)(v5 + 616) = v5 + 632;
    *(_QWORD *)(v5 + 280) = 0;
    *(_QWORD *)(v5 + 288) = 0;
    *(_QWORD *)(v5 + 296) = 0;
    *(_DWORD *)(v5 + 304) = 0;
    *(_QWORD *)(v5 + 312) = 0;
    *(_QWORD *)(v5 + 320) = 0;
    *(_QWORD *)(v5 + 328) = 0;
    *(_DWORD *)(v5 + 336) = 0;
    *(_QWORD *)(v5 + 352) = 0x400000000LL;
    *(_QWORD *)(v5 + 624) = 0x400000000LL;
    *(_QWORD *)(v5 + 696) = v5 + 712;
    *(_QWORD *)(v5 + 704) = 0x400000000LL;
    *(_QWORD *)(v5 + 968) = 0;
    *(_DWORD *)(v5 + 976) = 0;
LABEL_3:
    *a1 = v5;
    return a1;
  }
  if ( !v3 )
    goto LABEL_3;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 8LL))(v3);
  *a1 = 0;
  return a1;
}
