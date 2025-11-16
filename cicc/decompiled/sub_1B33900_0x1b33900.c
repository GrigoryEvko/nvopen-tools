// Function: sub_1B33900
// Address: 0x1b33900
//
void __fastcall sub_1B33900(__int64 a1, char a2)
{
  __int64 v2; // r12
  _QWORD *v3; // rax
  _QWORD *v4; // r14
  __int64 v5; // rbx
  _QWORD *v6; // rax
  _QWORD *v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rax

  v2 = *(_QWORD *)(a1 + 8);
  while ( v2 )
  {
    while ( 1 )
    {
      v3 = sub_1648700(v2);
      v2 = *(_QWORD *)(v2 + 8);
      v4 = v3;
      if ( (unsigned __int8)(*((_BYTE *)v3 + 16) - 54) > 1u )
      {
        if ( *(_BYTE *)(*v3 + 8LL) )
        {
          v5 = v3[1];
          while ( v5 )
          {
            v6 = sub_1648700(v5);
            v5 = *(_QWORD *)(v5 + 8);
            v7 = v6;
            if ( a2 )
            {
              if ( *((_BYTE *)v6 + 16) != 78 )
                continue;
              v8 = *(v6 - 3);
              if ( *(_BYTE *)(v8 + 16) || (*(_BYTE *)(v8 + 33) & 0x20) == 0 )
                continue;
            }
            sub_15F20C0(v7);
          }
        }
        if ( !a2 )
          break;
        if ( *((_BYTE *)v4 + 16) == 78 )
        {
          v9 = *(v4 - 3);
          if ( !*(_BYTE *)(v9 + 16) && (*(_BYTE *)(v9 + 33) & 0x20) != 0 )
            break;
        }
      }
      if ( !v2 )
        return;
    }
    sub_15F20C0(v4);
  }
}
