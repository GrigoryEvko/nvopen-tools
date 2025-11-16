// Function: sub_16D1820
// Address: 0x16d1820
//
void __fastcall sub_16D1820(unsigned __int8 *a1, __int64 a2, __int64 a3)
{
  unsigned __int8 *v3; // r13
  unsigned __int8 *v4; // rbx
  unsigned __int8 v6; // si
  unsigned __int8 *v7; // rax

  v3 = &a1[a2];
  if ( a1 != &a1[a2] )
  {
    v4 = a1;
    do
    {
      while ( 1 )
      {
        v6 = *v4;
        if ( (unsigned __int8)(*v4 - 65) < 0x1Au )
          v6 = *v4 + 32;
        v7 = *(unsigned __int8 **)(a3 + 24);
        if ( (unsigned __int64)v7 >= *(_QWORD *)(a3 + 16) )
          break;
        ++v4;
        *(_QWORD *)(a3 + 24) = v7 + 1;
        *v7 = v6;
        if ( v3 == v4 )
          return;
      }
      ++v4;
      sub_16E7DE0(a3, v6);
    }
    while ( v3 != v4 );
  }
}
