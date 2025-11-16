// Function: sub_AF2A20
// Address: 0xaf2a20
//
unsigned __int64 __fastcall sub_AF2A20(__int64 a1)
{
  unsigned __int8 v1; // al
  __int64 v2; // rdi
  unsigned __int64 v3; // rax
  int v4; // ecx

  v1 = *(_BYTE *)(a1 - 16);
  if ( (v1 & 2) != 0 )
    v2 = *(_QWORD *)(a1 - 32);
  else
    v2 = a1 - 16 - 8LL * ((v1 >> 2) & 0xF);
  v3 = *(_QWORD *)(v2 + 8);
  if ( !v3 )
    return 0;
  v4 = *(unsigned __int8 *)v3;
  if ( (unsigned int)(v4 - 25) <= 1 )
    return v3 & 0xFFFFFFFFFFFFFFFBLL;
  if ( (_BYTE)v4 == 7 )
    return v3 | 4;
  else
    return 0;
}
