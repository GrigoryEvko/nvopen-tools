// Function: sub_250D2C0
// Address: 0x250d2c0
//
unsigned __int64 __fastcall sub_250D2C0(unsigned __int64 a1, unsigned __int64 a2)
{
  int v2; // eax
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  unsigned __int64 v6[2]; // [rsp+0h] [rbp-10h] BYREF

  v2 = *(unsigned __int8 *)a1;
  if ( (_BYTE)v2 == 22 )
  {
    v6[1] = a2;
    nullsub_1518();
    return a1 & 0xFFFFFFFFFFFFFFFCLL;
  }
  else if ( (unsigned __int8)v2 > 0x1Cu
         && (v4 = (unsigned int)(v2 - 34), (unsigned __int8)v4 <= 0x33u)
         && (v5 = 0x8000000000041LL, _bittest64(&v5, v4)) )
  {
    sub_250D230(v6, a1, 3, 0);
    return v6[0];
  }
  else
  {
    sub_250D230(v6, a1, 1, a2);
    return v6[0];
  }
}
