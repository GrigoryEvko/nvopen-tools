// Function: sub_261A0B0
// Address: 0x261a0b0
//
void __fastcall sub_261A0B0(__int64 **a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v4; // rax
  _QWORD *v5; // r12
  __int64 v6; // rbx
  __int64 v7; // rax
  _QWORD *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rax

  v2 = *(_QWORD *)(a2 + 16);
  while ( v2 )
  {
    v4 = v2;
    v2 = *(_QWORD *)(v2 + 8);
    v5 = *(_QWORD **)(v4 + 24);
    v6 = v5[2];
    if ( v6 )
    {
      do
      {
        v7 = v6;
        v6 = *(_QWORD *)(v6 + 8);
        v8 = *(_QWORD **)(v7 + 24);
        if ( *(_BYTE *)v8 == 85 )
        {
          v10 = *(v8 - 4);
          if ( v10 )
          {
            if ( !*(_BYTE *)v10
              && *(_QWORD *)(v10 + 24) == v8[10]
              && (*(_BYTE *)(v10 + 33) & 0x20) != 0
              && *(_DWORD *)(v10 + 36) == 11 )
            {
              sub_B43D60(v8);
            }
          }
        }
      }
      while ( v6 );
      if ( v5[2] )
      {
        v9 = sub_ACD6D0(*a1);
        sub_BD84D0((__int64)v5, v9);
      }
    }
    sub_B43D60(v5);
  }
}
