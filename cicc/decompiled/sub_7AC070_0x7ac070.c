// Function: sub_7AC070
// Address: 0x7ac070
//
__int64 __fastcall sub_7AC070(__int64 a1, int a2)
{
  int v2; // ebx
  _BYTE *v3; // rax
  unsigned int v4; // r8d
  _QWORD key[3]; // [rsp+8h] [rbp-18h] BYREF

  key[0] = a1;
  if ( dword_4F077C4 == 2 )
  {
    LOBYTE(v2) = 8;
    if ( dword_4D0419C )
    {
      LOBYTE(v2) = 4;
      if ( unk_4F07778 <= 201102 )
        v2 = dword_4F07774 == 0 ? 1 : 4;
    }
  }
  else if ( unk_4F07778 <= 202310 )
  {
    LOBYTE(v2) = 4;
    if ( unk_4F07778 <= 201111 )
      LOBYTE(v2) = (unk_4F07778 > 199900) + 1;
  }
  else
  {
    v2 = dword_4D0419C == 0 ? 8 : 4;
  }
  v3 = bsearch(key, &unk_4B6F3E0, 0x7BEu, 0x18u, (__compar_fn_t)sub_7AB700);
  if ( !v3 )
    return 968;
  v4 = 968;
  if ( (v3[16] & (unsigned __int8)v2) != 0 )
  {
    v4 = 0;
    if ( a2 )
      return (v3[17] & (unsigned __int8)v2) != 0 ? 0x40D : 0;
  }
  return v4;
}
