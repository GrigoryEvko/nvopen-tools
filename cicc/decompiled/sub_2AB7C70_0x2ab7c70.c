// Function: sub_2AB7C70
// Address: 0x2ab7c70
//
__int64 __fastcall sub_2AB7C70(_BYTE *a1, __int16 a2)
{
  char v2; // dl
  __int64 result; // rax

  v2 = a2;
  if ( !(_BYTE)a2 )
    v2 = LOBYTE(qword_500D0A0[17]) ^ 1;
  result = HIBYTE(a2);
  *a1 = v2;
  if ( !HIBYTE(a2) )
    result = LOBYTE(qword_500CFC0[17]) ^ 1u;
  a1[1] = result;
  return result;
}
