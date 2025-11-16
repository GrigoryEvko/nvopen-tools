// Function: sub_CE3280
// Address: 0xce3280
//
__int64 __fastcall sub_CE3280(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rax
  __int64 v4; // r13
  int v5; // eax
  __int64 v6; // rdx

  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 96) = a1 + 112;
  *(_QWORD *)(a1 + 104) = 0x800000000LL;
  *(_QWORD *)(a1 + 16) = 0x100000008LL;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  v2 = *(_QWORD *)(a2 + 48);
  *(_QWORD *)(a1 + 32) = a2;
  v3 = v2 & 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)a1 = 1;
  if ( v3 == a2 + 48 )
    goto LABEL_6;
  if ( !v3 )
    BUG();
  v4 = v3 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v3 - 24) - 30 > 0xA )
  {
LABEL_6:
    v5 = 0;
    v6 = 0;
    v4 = 0;
  }
  else
  {
    v5 = sub_B46E30(v4);
    v6 = v4;
  }
  *(_QWORD *)(a1 + 128) = v4;
  *(_QWORD *)(a1 + 144) = a2;
  *(_QWORD *)(a1 + 112) = v6;
  *(_DWORD *)(a1 + 120) = v5;
  *(_DWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 104) = 1;
  return sub_CE27D0(a1);
}
