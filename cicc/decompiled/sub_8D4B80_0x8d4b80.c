// Function: sub_8D4B80
// Address: 0x8d4b80
//
__int64 __fastcall sub_8D4B80(__int64 a1)
{
  char v1; // al
  __int64 v2; // r8
  char v3; // al

  v1 = *(_BYTE *)(a1 + 140);
  v2 = a1;
  if ( v1 == 12 )
  {
    do
    {
      a1 = *(_QWORD *)(a1 + 160);
      v3 = *(_BYTE *)(a1 + 140);
    }
    while ( v3 == 12 );
    if ( (unsigned __int8)(v3 - 9) <= 2u && *(char *)(*(_QWORD *)(*(_QWORD *)a1 + 96LL) + 178LL) >= 0 )
      return sub_730E80(a1);
    return sub_8D4A00(v2);
  }
  else
  {
    if ( (unsigned __int8)(v1 - 9) <= 2u && *(char *)(*(_QWORD *)(*(_QWORD *)a1 + 96LL) + 178LL) >= 0 )
      return sub_730E80(a1);
    if ( dword_4F077C0 && (v1 == 1 || v1 == 7) )
      return 1;
    else
      return *(_QWORD *)(a1 + 128);
  }
}
