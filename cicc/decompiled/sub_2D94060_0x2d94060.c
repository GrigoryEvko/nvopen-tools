// Function: sub_2D94060
// Address: 0x2d94060
//
void __fastcall sub_2D94060(const void *a1, size_t a2, const void *a3, size_t a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v8; // r8
  __int64 i; // [rsp+8h] [rbp-38h]

  v5 = *(_QWORD *)(a5 + 32);
  for ( i = a5 + 24; i != v5; v5 = *(_QWORD *)(v5 + 8) )
  {
    v8 = v5 - 56;
    if ( !v5 )
      v8 = 0;
    sub_2D93070(a1, a2, a3, a4, v8);
  }
}
