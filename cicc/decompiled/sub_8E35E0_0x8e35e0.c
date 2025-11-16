// Function: sub_8E35E0
// Address: 0x8e35e0
//
__int64 __fastcall sub_8E35E0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi
  __int64 *v4; // rcx
  __int64 **v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rdx

  v3 = **(_QWORD ***)(a2 + 168);
  if ( v3 )
  {
    v4 = *(__int64 **)(a1 + 40);
    do
    {
      v5 = *(__int64 ***)(v3[5] + 168LL);
      while ( 1 )
      {
        v5 = (__int64 **)*v5;
        if ( !v5 )
          break;
        v6 = v5[5];
        if ( v4 != v6 )
        {
          if ( !v4 )
            continue;
          if ( !v6 )
            continue;
          if ( !dword_4F07588 )
            continue;
          v7 = v6[4];
          if ( v4[4] != v7 || !v7 )
            continue;
        }
        if ( ((_BYTE)v5[12] & 2) != 0 )
          return 1;
        break;
      }
      v3 = (_QWORD *)*v3;
    }
    while ( v3 );
  }
  return 0;
}
