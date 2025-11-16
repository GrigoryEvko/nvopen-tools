// Function: sub_937280
// Address: 0x937280
//
__int64 __fastcall sub_937280(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // r13d
  int v5; // ebx
  __int64 v6; // rax
  __int64 v7; // rax

  v4 = a3;
  if ( *(char *)(a2 + 142) >= 0 && *(_BYTE *)(a2 + 140) == 12 )
    v5 = sub_8D4AB0(a2);
  else
    v5 = *(_DWORD *)(a2 + 136);
  v6 = sub_91A3A0(*(_QWORD *)(a1 + 32) + 8LL, a2, a3, a4);
  if ( v5 == (unsigned int)(1LL << sub_AE5020(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 352LL), v6)) )
    return 0;
  v7 = sub_BCB2D0(*(_QWORD *)(a1 + 40));
  return sub_ACD640(v7, (v4 << 16) | (unsigned int)(unsigned __int16)v5, 0);
}
