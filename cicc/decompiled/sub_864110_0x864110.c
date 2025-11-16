// Function: sub_864110
// Address: 0x864110
//
_QWORD *__fastcall sub_864110(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v6; // rdi
  __int64 v7; // rbx
  int v8; // r13d
  _QWORD *v9; // r12
  _QWORD *result; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 *v15; // r9

  v6 = dword_4F04C64;
  v7 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v8 = *(_DWORD *)(v7 + 452);
  if ( v8 == -1 )
    goto LABEL_5;
  if ( *(_BYTE *)(v7 + 4) == 14 )
    sub_8845B0(dword_4F04C64);
  unk_4F04C2C = v8;
  if ( dword_4F077C4 == 2
    && (v6 = (int)dword_4F04C40,
        v11 = 776LL * (int)dword_4F04C40,
        *(_BYTE *)(qword_4F04C68[0] + v11 + 7) &= ~8u,
        a3 = qword_4F04C68[0],
        *(_QWORD *)(qword_4F04C68[0] + v11 + 456)) )
  {
    sub_8845B0(v6);
    if ( *(_BYTE *)(v7 + 4) == 14 )
      goto LABEL_11;
  }
  else
  {
LABEL_5:
    if ( *(_BYTE *)(v7 + 4) == 14 )
    {
LABEL_11:
      sub_863FC0(v6, a2, a3, a4, a5, a6);
      return sub_863FE0(v6, a2, v12, v13, v14, v15);
    }
  }
  v9 = *(_QWORD **)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 408);
  result = sub_863FE0(v6, a2, a3, a4, a5, a6);
  if ( v9 )
  {
    if ( !*v9 )
      return (_QWORD *)sub_878D40(v9);
  }
  return result;
}
