// Function: sub_2D69E90
// Address: 0x2d69e90
//
void __fastcall sub_2D69E90(__int64 a1)
{
  __int64 v1; // rsi
  _QWORD *v2; // rbx
  _QWORD *v3; // r13
  __int64 v4; // rsi

  sub_2D69B50(a1);
  if ( *(_BYTE *)(a1 + 64) )
  {
    v1 = *(unsigned int *)(a1 + 56);
    *(_BYTE *)(a1 + 64) = 0;
    if ( (_DWORD)v1 )
    {
      v2 = *(_QWORD **)(a1 + 40);
      v3 = &v2[2 * v1];
      do
      {
        if ( *v2 != -8192 && *v2 != -4096 )
        {
          v4 = v2[1];
          if ( v4 )
            sub_B91220((__int64)(v2 + 1), v4);
        }
        v2 += 2;
      }
      while ( v3 != v2 );
      v1 = *(unsigned int *)(a1 + 56);
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 40), 16 * v1, 8);
  }
}
