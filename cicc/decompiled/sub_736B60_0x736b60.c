// Function: sub_736B60
// Address: 0x736b60
//
__int64 __fastcall sub_736B60(__int64 a1, int a2, _QWORD *a3)
{
  char v5; // dl
  __int64 v6; // rbx
  __int64 v7; // rdi

  v5 = *(_BYTE *)(qword_4F04C68[0] + 776LL * a2 + 4);
  v6 = *(_QWORD *)a1;
  if ( v5 && (unsigned __int8)(v5 - 3) > 1u )
    sub_721090();
  v7 = *(_QWORD *)a1;
  *(_QWORD *)(v6 + 48) = *a3;
  sub_885FF0(v7, (unsigned int)a2, 0);
  sub_877D80(a1, v6);
  sub_877E90(v6, a1);
  *(_BYTE *)(a1 + 88) = (4 * (*(_BYTE *)(v6 + 81) & 1)) | *(_BYTE *)(a1 + 88) & 0xFB;
  return sub_7365B0(a1, a2);
}
