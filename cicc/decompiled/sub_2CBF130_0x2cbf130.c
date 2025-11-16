// Function: sub_2CBF130
// Address: 0x2cbf130
//
__int64 __fastcall sub_2CBF130(__int64 a1)
{
  __int64 v1; // rdx
  char v2; // cl
  __int64 v3; // rax
  unsigned int v4; // r8d

  v1 = *(_QWORD *)(a1 + 16);
  do
  {
    if ( !v1 )
      return 0;
    v2 = **(_BYTE **)(v1 + 24);
    v3 = v1;
    v1 = *(_QWORD *)(v1 + 8);
  }
  while ( (unsigned __int8)(v2 - 30) > 0xAu );
  v4 = 0;
LABEL_6:
  ++v4;
  while ( 1 )
  {
    v3 = *(_QWORD *)(v3 + 8);
    if ( !v3 )
      return v4;
    if ( (unsigned __int8)(**(_BYTE **)(v3 + 24) - 30) <= 0xAu )
      goto LABEL_6;
  }
}
