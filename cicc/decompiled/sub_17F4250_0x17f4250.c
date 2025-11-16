// Function: sub_17F4250
// Address: 0x17f4250
//
unsigned __int64 __fastcall sub_17F4250(unsigned __int64 a1)
{
  unsigned __int64 v1; // r12
  __int64 v2; // r10
  __int64 v3; // rbx
  char v4; // r10
  int v5; // r15d

  v1 = HIBYTE(a1);
  v2 = (unsigned int)(dword_4FA6740 - 1);
  if ( (unsigned int)v2 > 3 )
  {
    v5 = 0;
    v4 = 0;
    v3 = 0;
  }
  else
  {
    v3 = dword_42B70B0[v2];
    v4 = *((_BYTE *)&off_42B70A6 + v2);
    v5 = v3;
  }
  if ( (int)a1 >= v5 )
    v3 = (unsigned int)a1;
  LOBYTE(v1) = byte_4FA6200 | HIBYTE(a1);
  return ((unsigned __int64)(unsigned __int8)(byte_4FA62E0 | BYTE6(a1)) << 48) & 0xFFFFFFFFFFFFFFLL
       | ((unsigned __int64)(unsigned __int8)(BYTE4(a1) | v4) << 32) & 0xFFFFFFFFFFFFLL
       | v3 & 0xFF00FFFFFFFFLL
       | a1 & 0xFF0000000000LL
       | (v1 << 56);
}
