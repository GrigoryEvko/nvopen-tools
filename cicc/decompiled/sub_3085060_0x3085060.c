// Function: sub_3085060
// Address: 0x3085060
//
__int64 __fastcall sub_3085060(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned int v5; // r8d
  unsigned int v6; // edx
  __int64 v7; // rax
  _BYTE *v8; // r12
  unsigned int v10; // eax

  v2 = *(_QWORD *)(a1 + 40);
  if ( (_DWORD)v2 )
  {
    v3 = 0;
    v4 = 8LL * (unsigned int)v2;
    v5 = 0;
    LOBYTE(v6) = 0;
    while ( 1 )
    {
      v7 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + v3);
      if ( *(_WORD *)(v7 + 24) != 15 )
        goto LABEL_5;
      if ( (_BYTE)v6 )
        return 0;
      v8 = *(_BYTE **)(v7 - 8);
      if ( *v8 == 22 )
      {
        v6 = byte_502D428[0];
        if ( byte_502D428[0] )
        {
          *a2 = v8;
          v5 = v6;
        }
        else
        {
          v10 = sub_B2D700((__int64)v8);
          *a2 = v8;
          v5 = v10;
          LOBYTE(v6) = v10;
          if ( !(_BYTE)v10 )
            goto LABEL_10;
        }
LABEL_5:
        v3 += 8;
        if ( v4 == v3 )
          return v5;
      }
      else
      {
        v5 = 0;
LABEL_10:
        v3 += 8;
        LOBYTE(v6) = 1;
        if ( v4 == v3 )
          return v5;
      }
    }
  }
  return 0;
}
