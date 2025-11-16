// Function: sub_29C2140
// Address: 0x29c2140
//
void __fastcall sub_29C2140(__int64 a1, _QWORD *a2)
{
  __int64 *v2; // r12
  __int64 v3; // r13
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 *v7; // r12
  __int64 v8; // rbx
  __int64 v9; // rax

  v2 = *(__int64 **)a1;
  v3 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v3 )
  {
    do
    {
      if ( a2 )
      {
        v5 = *v2;
        a2[1] = 4;
        a2[2] = 0;
        *a2 = v5;
        v6 = v2[3];
        a2[3] = v6;
        if ( v6 != 0 && v6 != -4096 && v6 != -8192 )
          sub_BD6050(a2 + 1, v2[1] & 0xFFFFFFFFFFFFFFF8LL);
      }
      v2 += 4;
      a2 += 4;
    }
    while ( (__int64 *)v3 != v2 );
    v7 = *(__int64 **)a1;
    v8 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v8 )
    {
      do
      {
        v9 = *(_QWORD *)(v8 - 8);
        v8 -= 32;
        if ( v9 != 0 && v9 != -4096 && v9 != -8192 )
          sub_BD60C0((_QWORD *)(v8 + 8));
      }
      while ( (__int64 *)v8 != v7 );
    }
  }
}
