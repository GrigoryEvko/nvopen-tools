// Function: sub_28EEFB0
// Address: 0x28eefb0
//
void __fastcall sub_28EEFB0(__int64 a1)
{
  __int64 v1; // r12
  unsigned __int64 *v2; // rbx
  unsigned __int64 *v3; // r12
  __int64 v4; // rax
  bool v5; // zf
  _QWORD v6[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v7; // [rsp+10h] [rbp-20h]

  v1 = *(unsigned int *)(a1 + 24);
  v2 = *(unsigned __int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  v6[0] = 0;
  v6[1] = 0;
  v3 = &v2[4 * v1];
  v7 = -4096;
  if ( v3 != v2 )
  {
    do
    {
      if ( v2 )
      {
        *v2 = 0;
        v2[1] = 0;
        v4 = v7;
        v5 = v7 == 0;
        v2[2] = v7;
        if ( v4 != -4096 && !v5 && v4 != -8192 )
          sub_BD6050(v2, v6[0] & 0xFFFFFFFFFFFFFFF8LL);
      }
      v2 += 4;
    }
    while ( v2 != v3 );
    if ( v7 != 0 && v7 != -4096 && v7 != -8192 )
      sub_BD60C0(v6);
  }
}
