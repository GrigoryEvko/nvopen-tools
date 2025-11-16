// Function: sub_19A22F0
// Address: 0x19a22f0
//
void __fastcall sub_19A22F0(__int64 *a1, __int64 a2, unsigned int a3, __int64 a4, int a5, __m128i a6, __m128i a7)
{
  __int64 v9; // rbx
  __int64 v10; // r9
  __int64 v11; // [rsp+0h] [rbp-40h]
  int v12; // [rsp+Ch] [rbp-34h]

  v11 = *(unsigned int *)(a4 + 40);
  if ( *(_DWORD *)(a4 + 40) )
  {
    v9 = 0;
    do
    {
      v10 = v9++;
      v12 = a5;
      sub_19A1B20(a1, a2, a3, a4, a5, v10, a6, a7, 0);
      a5 = v12;
    }
    while ( v11 != v9 );
  }
  if ( *(_QWORD *)(a4 + 24) == 1 )
    sub_19A1B20(a1, a2, a3, a4, a5, -1, a6, a7, 1);
}
