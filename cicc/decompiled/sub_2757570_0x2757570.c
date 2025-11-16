// Function: sub_2757570
// Address: 0x2757570
//
void __fastcall sub_2757570(__int64 a1)
{
  _QWORD *v1; // r13
  _QWORD *v2; // r12
  __int64 v3; // rax
  _QWORD *v4; // r13
  _QWORD *v5; // r12
  __int64 v6; // rax

  sub_2753F50(*(_QWORD **)(a1 + 728));
  v1 = *(_QWORD **)(a1 + 504);
  v2 = &v1[3 * *(unsigned int *)(a1 + 512)];
  if ( v1 != v2 )
  {
    do
    {
      v3 = *(v2 - 1);
      v2 -= 3;
      if ( v3 != 0 && v3 != -4096 && v3 != -8192 )
        sub_BD60C0(v2);
    }
    while ( v1 != v2 );
    v2 = *(_QWORD **)(a1 + 504);
  }
  if ( v2 != (_QWORD *)(a1 + 520) )
    _libc_free((unsigned __int64)v2);
  if ( !*(_BYTE *)(a1 + 436) )
    _libc_free(*(_QWORD *)(a1 + 416));
  v4 = *(_QWORD **)(a1 + 8);
  v5 = &v4[3 * *(unsigned int *)(a1 + 16)];
  if ( v4 != v5 )
  {
    do
    {
      v6 = *(v5 - 1);
      v5 -= 3;
      if ( v6 != -4096 && v6 != 0 && v6 != -8192 )
        sub_BD60C0(v5);
    }
    while ( v4 != v5 );
    v5 = *(_QWORD **)(a1 + 8);
  }
  if ( v5 != (_QWORD *)(a1 + 24) )
    _libc_free((unsigned __int64)v5);
}
