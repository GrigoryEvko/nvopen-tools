// Function: sub_26476C0
// Address: 0x26476c0
//
__int64 __fastcall sub_26476C0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 *v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rax
  volatile signed __int32 *v8; // rdi
  volatile signed __int32 *v9; // rdi

  v3 = *(_QWORD *)(a1 + 8);
  v4 = (__int64 *)(a2 + 16);
  if ( a2 + 16 != v3 )
  {
    v5 = (v3 - (__int64)v4) >> 4;
    if ( v3 - (__int64)v4 > 0 )
    {
      do
      {
        v6 = *v4;
        v7 = v4[1];
        *v4 = 0;
        v8 = (volatile signed __int32 *)*(v4 - 1);
        v4[1] = 0;
        *(v4 - 2) = v6;
        *(v4 - 1) = v7;
        if ( v8 )
          sub_A191D0(v8);
        v4 += 2;
        --v5;
      }
      while ( v5 );
      v3 = *(_QWORD *)(a1 + 8);
    }
  }
  *(_QWORD *)(a1 + 8) = v3 - 16;
  v9 = *(volatile signed __int32 **)(v3 - 8);
  if ( v9 )
    sub_A191D0(v9);
  return a2;
}
