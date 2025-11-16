// Function: sub_8D4AB0
// Address: 0x8d4ab0
//
__int64 __fastcall sub_8D4AB0(__int64 a1)
{
  __int64 v1; // rdx
  int v2; // ecx
  unsigned int v3; // eax
  unsigned int v4; // r12d
  char v5; // al
  unsigned __int64 v6; // rax

  if ( (*(_DWORD *)(a1 + 140) & 0x8000FF) != 0xC )
    return *(unsigned int *)(a1 + 136);
  v1 = a1;
  v2 = 0;
  do
  {
    v3 = *(unsigned __int8 *)(v1 + 185);
    v1 = *(_QWORD *)(v1 + 160);
    v2 |= (v3 >> 3) & 1;
  }
  while ( (*(_DWORD *)(v1 + 140) & 0x8000FF) == 0xC );
  v4 = *(_DWORD *)(v1 + 136);
  if ( v2 && (_DWORD)qword_4F077B4 )
  {
    v5 = *(_BYTE *)(a1 + 140);
    if ( v5 == 12 )
    {
      v6 = sub_8D4A00(a1);
    }
    else if ( dword_4F077C0 && (v5 == 1 || v5 == 7) )
    {
      v6 = 1;
    }
    else
    {
      v6 = *(_QWORD *)(a1 + 128);
    }
    if ( v6 <= unk_4F06AD8 && v6 > v4 )
      return (unsigned int)v6;
  }
  return v4;
}
