// Function: sub_C35610
// Address: 0xc35610
//
__int64 __fastcall sub_C35610(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // rbx
  char v4; // al

  v2 = *(_BYTE *)(a2 + 20) & 7;
  if ( v2 == 1 )
  {
    LOBYTE(v3) = (*(_QWORD *)a2 != (_QWORD)&unk_3F65660) - 1;
    sub_C33930(a2);
LABEL_8:
    v3 = (unsigned __int8)v3;
    goto LABEL_9;
  }
  if ( v2 == 3 || !v2 )
    BUG();
  LODWORD(v3) = *(_DWORD *)(a2 + 16) + (*(_QWORD *)a2 != (_QWORD)&unk_3F65660) + 127;
  sub_C33930(a2);
  if ( (_DWORD)v3 != 1 )
    goto LABEL_8;
  v3 = *(_QWORD *)sub_C33930(a2) & 1LL;
LABEL_9:
  v4 = *(_BYTE *)(a2 + 20);
  *(_DWORD *)(a1 + 8) = 8;
  *(_QWORD *)a1 = v3 | ((unsigned __int64)((v4 & 8) != 0) << 7);
  return a1;
}
