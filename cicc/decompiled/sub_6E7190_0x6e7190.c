// Function: sub_6E7190
// Address: 0x6e7190
//
__int64 __fastcall sub_6E7190(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 result; // rax

  sub_6E2E50(3, a3);
  *(_BYTE *)(a3 + 17) = 3;
  if ( dword_4F04C44 == -1 )
  {
    v5 = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
    if ( (v5[6] & 6) == 0 && v5[4] != 12 )
      goto LABEL_10;
  }
  else
  {
    v5 = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
  }
  if ( (v5[12] & 0x10) == 0 && a2 )
  {
    v6 = *(_QWORD *)(a2 + 40);
    if ( v6 && (unsigned int)sub_89A370(v6) )
      v7 = *(_QWORD *)&dword_4D03B80;
    else
      v7 = sub_72CBA0();
    *(_QWORD *)a3 = v7;
    *(_QWORD *)(a3 + 136) = a1;
    *(_QWORD *)(a3 + 68) = *(_QWORD *)&dword_4F063F8;
    *(_QWORD *)(a3 + 76) = qword_4F063F0;
    goto LABEL_14;
  }
LABEL_10:
  v8 = sub_72CBA0();
  *(_QWORD *)(a3 + 136) = a1;
  *(_QWORD *)a3 = v8;
  result = *(_QWORD *)&dword_4F063F8;
  *(_QWORD *)(a3 + 76) = qword_4F063F0;
  *(_QWORD *)(a3 + 68) = result;
  if ( !a2 )
  {
    *(_QWORD *)(a3 + 112) = result;
    return result;
  }
LABEL_14:
  *(_BYTE *)(a3 + 19) = (8 * (*(_BYTE *)(a2 + 18) & 1)) | *(_BYTE *)(a3 + 19) & 0xF7;
  *(_QWORD *)(a3 + 104) = *(_QWORD *)(a2 + 40);
  return sub_6E46C0(a3, a2);
}
