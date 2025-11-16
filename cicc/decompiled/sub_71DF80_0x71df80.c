// Function: sub_71DF80
// Address: 0x71df80
//
__int64 __fastcall sub_71DF80(__int64 a1)
{
  __int64 v1; // r15
  __int64 v2; // rbx
  __int64 v3; // r13
  _QWORD *v5; // r14
  char v6; // al

  v1 = unk_4D03B98;
  v2 = *(_QWORD *)(sub_72B840(a1) + 80);
  if ( !v2 )
    v2 = *(_QWORD *)(v1 + 8);
  if ( *(_BYTE *)(v2 + 40) == 19 )
  {
    v1 += 176;
    v2 = *(_QWORD *)(*(_QWORD *)(v2 + 72) + 8LL);
  }
  if ( *(char *)(a1 + 207) < 0 )
    return *(_QWORD *)(*(_QWORD *)(v2 + 72) + 72LL);
  v5 = (_QWORD *)sub_726B30(9);
  v3 = sub_726850();
  *(_QWORD *)(v3 + 112) = *(_QWORD *)(a1 + 64);
  v5[9] = v3;
  v5[3] = v2;
  v5[2] = *(_QWORD *)(v2 + 72);
  *(_QWORD *)(v2 + 72) = v5;
  if ( !v5[2] )
    *(_QWORD *)(v1 + 56) = v5;
  v6 = *(_BYTE *)(a1 + 193);
  *(_BYTE *)(a1 + 207) |= 0x80u;
  if ( (v6 & 5) == 0 )
    *(_BYTE *)(a1 + 193) = v6 & 0xFD;
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 174) - 1) <= 1u )
  {
    sub_6851C0(0xA6Bu, (_DWORD *)(v3 + 112));
    *(_BYTE *)(v3 + 120) |= 1u;
  }
  else if ( unk_4F07290 == a1 )
  {
    sub_6854C0(0xA6Cu, (FILE *)(v3 + 112), *(_QWORD *)a1);
    *(_BYTE *)(v3 + 120) |= 1u;
  }
  sub_87BD00(a1, v3);
  return v3;
}
