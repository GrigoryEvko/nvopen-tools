// Function: sub_22C3BD0
// Address: 0x22c3bd0
//
void __fastcall sub_22C3BD0(__int64 a1)
{
  bool v1; // zf
  unsigned __int64 *v2; // rbx
  __int64 v3; // r12
  unsigned __int64 *v4; // r12
  __int64 v5; // rax
  _QWORD v6[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v7; // [rsp+10h] [rbp-20h]

  v1 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v6[0] = 0;
  v6[1] = 0;
  v7 = -4096;
  if ( v1 )
  {
    v2 = *(unsigned __int64 **)(a1 + 16);
    v3 = 3LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v2 = (unsigned __int64 *)(a1 + 16);
    v3 = 6;
  }
  v4 = &v2[v3];
  if ( v4 != v2 )
  {
    do
    {
      if ( v2 )
      {
        *v2 = 0;
        v2[1] = 0;
        v5 = v7;
        v1 = v7 == 0;
        v2[2] = v7;
        if ( v5 != -4096 && !v1 && v5 != -8192 )
          sub_BD6050(v2, v6[0] & 0xFFFFFFFFFFFFFFF8LL);
      }
      v2 += 3;
    }
    while ( v2 != v4 );
    if ( v7 != -4096 && v7 != 0 && v7 != -8192 )
      sub_BD60C0(v6);
  }
}
