// Function: sub_86B430
// Address: 0x86b430
//
_QWORD *__fastcall sub_86B430(__int64 a1)
{
  __int64 v2; // rax
  __int64 *v3; // rdx
  __int64 v4; // r13
  __int64 v5; // rdx

  if ( dword_4F077C4 != 2 )
    return (_QWORD *)sub_8699D0(a1, 21, 0);
  v2 = unk_4D03B98 + 176LL * unk_4D03B90;
  v3 = *(__int64 **)(v2 + 144);
  if ( !v3 )
  {
    v4 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 328);
    if ( v4 )
      goto LABEL_5;
    return (_QWORD *)sub_8699D0(a1, 21, 0);
  }
  v4 = *v3;
  if ( !*v3 )
  {
    v5 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( *(_BYTE *)(v5 + 4) == 15 && (unsigned int)(*(_DWORD *)v2 - 1) <= 2 )
    {
      v4 = *(_QWORD *)(v5 + 328);
      goto LABEL_5;
    }
    return (_QWORD *)sub_8699D0(a1, 21, 0);
  }
LABEL_5:
  sub_8699D0(a1, 21, 0);
  return sub_86A220(*(__int64 **)(a1 + 56), dword_4F04C64, v4, dword_4F04C64);
}
