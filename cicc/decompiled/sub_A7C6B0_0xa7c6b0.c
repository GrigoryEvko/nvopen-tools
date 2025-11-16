// Function: sub_A7C6B0
// Address: 0xa7c6b0
//
bool __fastcall sub_A7C6B0(_BYTE *a1, __int64 a2)
{
  unsigned __int8 v2; // al
  _BYTE **v3; // rdi
  _BYTE *v4; // rdi
  _QWORD *v5; // rcx
  bool result; // al
  unsigned __int64 v7; // rdx

  if ( !a1 || *a1 != 5 )
    return 0;
  v2 = *(a1 - 16);
  if ( (v2 & 2) == 0 )
  {
    if ( (*((_WORD *)a1 - 8) & 0x3C0) != 0 )
    {
      v3 = (_BYTE **)&a1[-8 * ((v2 >> 2) & 0xF) - 16];
      goto LABEL_6;
    }
    return 0;
  }
  if ( !*((_DWORD *)a1 - 6) )
    return 0;
  v3 = (_BYTE **)*((_QWORD *)a1 - 4);
LABEL_6:
  v4 = *v3;
  if ( !v4 || *v4 )
    return 0;
  v5 = (_QWORD *)sub_B91420(v4, a2);
  result = 0;
  if ( v7 > 0xF )
    return (*v5 ^ 0x6365762E6D766C6CLL | v5[1] ^ 0x2E72657A69726F74LL) == 0;
  return result;
}
