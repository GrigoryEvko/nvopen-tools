// Function: sub_1F46A10
// Address: 0x1f46a10
//
__int64 (*__fastcall sub_1F46A10(_BYTE *a1))(void)
{
  __int64 (*result)(void); // rax
  __int64 v2; // rax
  _QWORD *v3; // r12
  __int64 v4; // rax
  _QWORD *v5; // rax

  result = *(__int64 (**)(void))(*(_QWORD *)a1 + 280LL);
  if ( result != sub_1F44630 )
    result = (__int64 (*)(void))result();
  if ( !a1[227] )
  {
    if ( a1[225] )
      return result;
LABEL_9:
    v5 = (_QWORD *)sub_1654860(1);
    return (__int64 (*)(void))sub_1F46490((__int64)a1, v5, 1, 1, 0);
  }
  v2 = sub_22077B0(160);
  v3 = (_QWORD *)v2;
  if ( v2 )
  {
    *(_QWORD *)(v2 + 8) = 0;
    *(_DWORD *)(v2 + 24) = 4;
    *(_QWORD *)(v2 + 16) = &unk_50516DC;
    *(_QWORD *)(v2 + 80) = v2 + 64;
    *(_QWORD *)(v2 + 88) = v2 + 64;
    *(_QWORD *)(v2 + 128) = v2 + 112;
    *(_QWORD *)(v2 + 136) = v2 + 112;
    *(_QWORD *)(v2 + 32) = 0;
    *(_QWORD *)(v2 + 40) = 0;
    *(_QWORD *)(v2 + 48) = 0;
    *(_DWORD *)(v2 + 64) = 0;
    *(_QWORD *)(v2 + 72) = 0;
    *(_QWORD *)(v2 + 96) = 0;
    *(_DWORD *)(v2 + 112) = 0;
    *(_QWORD *)(v2 + 120) = 0;
    *(_QWORD *)(v2 + 144) = 0;
    *(_BYTE *)(v2 + 152) = 0;
    *(_QWORD *)v2 = &unk_49FF230;
    v4 = sub_163A1D0();
    sub_384D3D0(v4);
  }
  result = (__int64 (*)(void))sub_1F46490((__int64)a1, v3, 1, 1, 0);
  if ( !a1[225] )
    goto LABEL_9;
  return result;
}
