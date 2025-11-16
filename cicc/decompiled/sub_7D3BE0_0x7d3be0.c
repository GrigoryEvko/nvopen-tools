// Function: sub_7D3BE0
// Address: 0x7d3be0
//
__int64 sub_7D3BE0()
{
  __int64 v1; // rdx
  __int64 v2; // rax

  if ( *(int *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 200) > 0 )
    return 0;
  v1 = 776LL * unk_4F04C48 + qword_4F04C68[0];
  if ( (*(_BYTE *)(v1 + 12) & 0x10) != 0 && (v2 = *(_QWORD *)(v1 + 368)) != 0 )
    return (unsigned int)(*(_DWORD *)(v2 + 44) - 1);
  else
    return *(unsigned int *)(*(_QWORD *)(v1 + 408) + 44LL);
}
