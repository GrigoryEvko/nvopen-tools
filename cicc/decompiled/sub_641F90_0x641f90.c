// Function: sub_641F90
// Address: 0x641f90
//
void __fastcall sub_641F90(__int64 *a1, __int64 a2)
{
  __int64 v3; // rdi
  char v4; // r13
  char v5; // r13
  _BYTE v6[36]; // [rsp+Ch] [rbp-24h] BYREF

  v3 = *a1;
  if ( ((*(_BYTE *)(v3 + 81) & 0x10) != 0 || !*(_QWORD *)(v3 + 64))
    && *(_DWORD *)(v3 + 40) != *(_DWORD *)qword_4F04C68[0]
    && (dword_4F04C58 & (unsigned int)sub_85EBD0(v3, v6)) == 0xFFFFFFFF )
  {
    *a1 = 0;
    v4 = *((_BYTE *)a1 + 88) >> 2;
    sub_877D80(a1, a2);
    v5 = a1[11] & 0xFB | (4 * (v4 & 1));
    *((_BYTE *)a1 + 88) = v5;
    if ( dword_4F077C4 == 2 && (v5 & 0x70) == 0x30 )
    {
      if ( (*(_BYTE *)(a2 + 81) & 0x10) != 0 || !*(_QWORD *)(a2 + 64) )
        a1[5] = unk_4F07288;
      else
        sub_877E90(0, a1);
    }
  }
}
