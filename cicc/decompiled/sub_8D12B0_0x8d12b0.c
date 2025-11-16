// Function: sub_8D12B0
// Address: 0x8d12b0
//
__int64 __fastcall sub_8D12B0(__int64 a1)
{
  char v1; // al
  unsigned __int8 v2; // dl
  int v3; // ecx

  v1 = *(_BYTE *)(a1 + 140);
  if ( (unsigned __int8)(v1 - 9) > 2u )
  {
    v3 = 0;
    v2 = 0;
    if ( v1 != 2 )
      goto LABEL_5;
    v2 = *(_BYTE *)(a1 + 161) & 8;
    if ( !v2 )
      goto LABEL_5;
    v2 = *(_BYTE *)(a1 + 163) & 7;
  }
  else
  {
    v2 = *(_BYTE *)(*(_QWORD *)(a1 + 168) + 109LL) & 7;
  }
  if ( v2 > 4u )
    goto LABEL_12;
  v3 = dword_3C22DF0[v2];
LABEL_5:
  if ( (unsigned __int8)byte_4F6055C > 4u )
LABEL_12:
    sub_721090();
  if ( dword_3C22DF0[(unsigned __int8)byte_4F6055C] < v3 )
    byte_4F6055C = v2;
  return 0;
}
