// Function: sub_688B20
// Address: 0x688b20
//
_DWORD *__fastcall sub_688B20(__int64 a1, __int64 a2)
{
  __int64 i; // rax
  char v3; // dl
  int v4; // ebx
  __int64 v5; // r12
  __int64 v6; // rax
  _DWORD *result; // rax

  for ( i = qword_4F04C68[0] + 776LL * dword_4F04C64; *(_BYTE *)(i + 4) != 1; i -= 776 )
    ;
  v3 = 1;
  v4 = ((*(_BYTE *)(i + 11) >> 6) ^ 1) & 1;
  while ( 1 )
  {
    if ( *(_DWORD *)i != *(_DWORD *)(a2 + 40) )
    {
      v4 += v3 == 1;
      goto LABEL_5;
    }
    if ( v3 == 1 )
      break;
LABEL_5:
    v3 = *(_BYTE *)(i - 772);
    i -= 776;
  }
  v5 = sub_726700(24);
  v6 = *(_QWORD *)(*(_QWORD *)(a2 + 88) + 16LL);
  *(_BYTE *)(v5 + 25) |= 1u;
  *(_QWORD *)v5 = v6;
  LODWORD(v6) = *(_DWORD *)(*(_QWORD *)(a2 + 88) + 120LL);
  *(_DWORD *)(v5 + 60) = v4;
  *(_DWORD *)(v5 + 56) = v6;
  sub_6E7150(v5, a1);
  result = &dword_4F077C4;
  *(_BYTE *)(a1 + 18) |= 8u;
  if ( dword_4F077C4 == 2 )
  {
    result = (_DWORD *)sub_8D32E0(*(_QWORD *)v5);
    if ( (_DWORD)result )
    {
      result = (_DWORD *)sub_6F82C0(a1);
      *(_BYTE *)(a1 + 18) |= 8u;
    }
  }
  return result;
}
