// Function: sub_234B9B0
// Address: 0x234b9b0
//
__int64 __fastcall sub_234B9B0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r14
  __int64 v3; // r13
  _QWORD *v4; // r15
  _QWORD *v5; // r12
  __int64 v6; // rax

  v1 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v1 )
  {
    v2 = *(_QWORD *)(a1 + 8);
    v3 = v2 + 72 * v1;
    do
    {
      if ( *(_QWORD *)v2 != -8192 && *(_QWORD *)v2 != -4096 )
      {
        v4 = *(_QWORD **)(v2 + 8);
        v5 = &v4[3 * *(unsigned int *)(v2 + 16)];
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
          v5 = *(_QWORD **)(v2 + 8);
        }
        if ( v5 != (_QWORD *)(v2 + 24) )
          _libc_free((unsigned __int64)v5);
      }
      v2 += 72;
    }
    while ( v3 != v2 );
    v1 = *(unsigned int *)(a1 + 24);
  }
  return sub_C7D6A0(*(_QWORD *)(a1 + 8), 72 * v1, 8);
}
