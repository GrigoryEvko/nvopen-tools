// Function: sub_19E3350
// Address: 0x19e3350
//
void __fastcall sub_19E3350(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rdx
  char *v4; // r14
  __int64 v5; // rbx
  unsigned __int64 v7; // rax
  char *v8; // rbx
  char *v9; // rdi

  v3 = 2 * a3;
  v4 = (char *)&a2[v3];
  if ( &a2[v3] != a2 )
  {
    v5 = v3 * 8;
    _BitScanReverse64(&v7, (v3 * 8) >> 4);
    sub_19E2E90(a2, (char *)&a2[v3], 2LL * (int)(63 - (v7 ^ 0x3F)), a1);
    if ( v5 <= 256 )
    {
      sub_19E2090((char *)a2, v4, a1);
    }
    else
    {
      v8 = (char *)(a2 + 32);
      sub_19E2090((char *)a2, (char *)a2 + 256, a1);
      if ( v4 != (char *)(a2 + 32) )
      {
        do
        {
          v9 = v8;
          v8 += 16;
          sub_19E1F60(v9, a1);
        }
        while ( v4 != v8 );
      }
    }
  }
}
