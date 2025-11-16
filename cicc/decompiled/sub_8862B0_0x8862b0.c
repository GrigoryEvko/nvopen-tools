// Function: sub_8862B0
// Address: 0x8862b0
//
_QWORD *__fastcall sub_8862B0(unsigned __int8 a1, __int64 a2)
{
  __int64 v2; // rbx
  _QWORD *v3; // r12
  __int64 v5; // rdx
  _DWORD v6[9]; // [rsp+Ch] [rbp-24h] BYREF

  v2 = *(_QWORD *)a2;
  v3 = sub_87EBB0(a1, *(_QWORD *)a2, (_QWORD *)(a2 + 8));
  *((_DWORD *)v3 + 10) = unk_4F066A8;
  if ( (*(_BYTE *)(a2 + 17) & 0x20) != 0 )
  {
    *((_BYTE *)v3 + 81) |= 0x20u;
    if ( dword_4F077C4 != 2 )
      goto LABEL_3;
  }
  else
  {
    v3[1] = *(_QWORD *)(v2 + 40);
    *(_QWORD *)(v2 + 40) = v3;
    if ( dword_4F077C4 != 2 )
      goto LABEL_3;
  }
  if ( (*(_BYTE *)(a2 + 18) & 2) == 0 && (v5 = *(_QWORD *)(a2 + 32)) != 0
    || dword_4F04C34 && (v5 = *(_QWORD *)(*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C34 + 184) + 32LL)) != 0 )
  {
    sub_877E90((__int64)v3, 0, v5);
  }
LABEL_3:
  sub_885620((__int64)v3, 0, v6);
  return v3;
}
