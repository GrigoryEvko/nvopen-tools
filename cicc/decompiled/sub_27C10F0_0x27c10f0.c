// Function: sub_27C10F0
// Address: 0x27c10f0
//
bool __fastcall sub_27C10F0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  _BYTE *v3; // rdx
  bool result; // al

  v2 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v2 == a2 + 48 )
    goto LABEL_8;
  if ( !v2 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v2 - 24) - 30 > 0xA )
LABEL_8:
    BUG();
  v3 = *(_BYTE **)(v2 - 120);
  result = 0;
  if ( *v3 == 82 )
  {
    result = 1;
    if ( a1 != *((_QWORD *)v3 - 8) )
      return *((_QWORD *)v3 - 4) == a1;
  }
  return result;
}
