// Function: sub_1314190
// Address: 0x1314190
//
void __fastcall sub_1314190(_QWORD *a1, __int64 *a2, unsigned int a3, __int64 a4)
{
  __int64 v4; // r9
  _QWORD *v5; // r8
  unsigned __int64 v6; // rbx
  unsigned int v8; // r12d
  unsigned int v9; // r13d
  __int64 i; // r15
  unsigned int v12; // eax
  __int64 v13; // r11
  unsigned int v14; // ecx
  __int64 v15; // rdi
  unsigned int v16; // ecx
  bool v17; // zf
  __int64 v18; // rax
  __int64 v19; // rdx

  v4 = a3;
  v5 = a1;
  v6 = a1[8];
  if ( a3 )
  {
    v8 = 0;
    v9 = 0;
    do
    {
      for ( i = v9; !v6; ++v9 )
      {
        i = v9 + 1;
        v6 = a1[i + 8];
      }
      v12 = sub_39FAC40(v6);
      v13 = a1[1];
      v14 = a3 - v8;
      if ( a3 - v8 >= v12 )
        v14 = v12;
      v15 = *a2;
      v16 = v8 + v14;
      do
      {
        v17 = !_BitScanForward64((unsigned __int64 *)&v18, v6);
        v19 = v8;
        if ( v17 )
          LODWORD(v18) = -1;
        ++v8;
        v6 ^= 1LL << v18;
        *(_QWORD *)(a4 + 8 * v19) = v13 + v15 * ((v9 << 6) + (__int64)(int)v18);
      }
      while ( v8 != v16 );
      a1[i + 8] = v6;
    }
    while ( a3 > v8 );
    v4 = a3;
    v5 = a1;
  }
  *v5 -= v4 << 28;
}
