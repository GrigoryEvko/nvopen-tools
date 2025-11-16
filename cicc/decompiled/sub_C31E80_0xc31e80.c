// Function: sub_C31E80
// Address: 0xc31e80
//
void __fastcall sub_C31E80(char *a1, unsigned int a2, unsigned int a3)
{
  __int64 v3; // rbx
  unsigned int v4; // r13d
  __int64 v5; // r14
  char v6; // cl

  v3 = a3;
  if ( a3 > 0x40 )
  {
    v4 = (a3 - 65) >> 6;
    v5 = v4 + 1;
    LOBYTE(v3) = a3 - ((_BYTE)v4 << 6) - 64;
    a1 = (char *)memset(a1, 255, 8 * v5);
LABEL_3:
    v6 = 64 - v3;
    v3 = (unsigned int)(v5 + 1);
    *(_QWORD *)&a1[8 * v5] = 0xFFFFFFFFFFFFFFFFLL >> v6;
    goto LABEL_4;
  }
  if ( a3 )
  {
    v5 = 0;
    goto LABEL_3;
  }
LABEL_4:
  if ( a2 > (unsigned int)v3 )
    memset(&a1[8 * v3], 0, 8LL * (a2 - 1 - (unsigned int)v3) + 8);
}
