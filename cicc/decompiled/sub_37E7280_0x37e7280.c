// Function: sub_37E7280
// Address: 0x37e7280
//
__int64 sub_37E7280()
{
  __int64 result; // rax
  _QWORD *v1; // rdx

  result = sub_22077B0(0x218u);
  if ( result )
  {
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)(result + 16) = &unk_505132C;
    *(_QWORD *)(result + 56) = result + 104;
    *(_DWORD *)(result + 24) = 2;
    *(_QWORD *)(result + 32) = 0;
    *(_QWORD *)(result + 40) = 0;
    *(_QWORD *)(result + 48) = 0;
    *(_QWORD *)(result + 64) = 1;
    *(_QWORD *)(result + 72) = 0;
    *(_QWORD *)(result + 80) = 0;
    *(_QWORD *)(result + 96) = 0;
    *(_QWORD *)(result + 104) = 0;
    *(_QWORD *)(result + 120) = 1;
    *(_QWORD *)(result + 128) = 0;
    *(_QWORD *)(result + 136) = 0;
    *(_QWORD *)(result + 152) = 0;
    *(_QWORD *)(result + 160) = 0;
    *(_BYTE *)(result + 168) = 0;
    *(_QWORD *)(result + 176) = 0;
    *(_QWORD *)(result + 184) = 0;
    *(_QWORD *)(result + 192) = 0;
    *(_QWORD *)result = off_4A3D538;
    *(_QWORD *)(result + 208) = 0x1000000000LL;
    *(_QWORD *)(result + 344) = 0;
    *(_QWORD *)(result + 352) = 0;
    *(_QWORD *)(result + 360) = 1;
    *(_QWORD *)(result + 112) = result + 160;
    *(_QWORD *)(result + 200) = result + 216;
    v1 = (_QWORD *)(result + 368);
    *(_DWORD *)(result + 88) = 1065353216;
    *(_DWORD *)(result + 144) = 1065353216;
    do
    {
      if ( v1 )
      {
        *v1 = -4096;
        v1[1] = -4096;
      }
      v1 += 2;
    }
    while ( v1 != (_QWORD *)(result + 432) );
    *(_QWORD *)(result + 432) = 0;
    *(_QWORD *)(result + 440) = 0;
    *(_QWORD *)(result + 448) = result + 472;
    *(_QWORD *)(result + 456) = 0;
    *(_QWORD *)(result + 464) = 8;
    *(_QWORD *)(result + 488) = 0;
    *(_DWORD *)(result + 496) = 0;
    *(_QWORD *)(result + 504) = 0;
    *(_QWORD *)(result + 512) = 0;
    *(_QWORD *)(result + 520) = 0;
    *(_QWORD *)(result + 528) = 0;
  }
  return result;
}
