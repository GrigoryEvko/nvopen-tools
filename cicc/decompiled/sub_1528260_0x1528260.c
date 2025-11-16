// Function: sub_1528260
// Address: 0x1528260
//
void __fastcall sub_1528260(_DWORD *a1, unsigned int a2, __int64 a3, unsigned int a4)
{
  unsigned int v5; // r15d
  __int64 v6; // rdx
  __int64 v7; // rbx
  unsigned int v8; // esi
  unsigned int v9; // [rsp+8h] [rbp-38h] BYREF
  char v10; // [rsp+Ch] [rbp-34h]

  v5 = *(_DWORD *)(a3 + 8);
  if ( a4 )
  {
    v6 = *(_QWORD *)a3;
    v10 = 1;
    v9 = a2;
    sub_1527BB0((__int64)a1, a4, v6, v5, 0, 0, (__int64)&v9);
  }
  else
  {
    v7 = 0;
    sub_1524D80(a1, 3u, a1[4]);
    sub_1524E40(a1, a2, 6);
    sub_1524E40(a1, v5, 6);
    if ( v5 )
    {
      do
      {
        v8 = *(_DWORD *)(*(_QWORD *)a3 + v7);
        v7 += 4;
        sub_1524E40(a1, v8, 6);
      }
      while ( 4LL * v5 != v7 );
    }
  }
}
