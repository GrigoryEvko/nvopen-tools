// Function: sub_1A689D0
// Address: 0x1a689d0
//
__int64 __fastcall sub_1A689D0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // r14
  __int64 v4; // r15
  __int64 v5; // rcx
  int v6; // r8d
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rsi

  v2 = sub_157EBA0(a2);
  if ( *(_BYTE *)(v2 + 16) != 26 )
    return 0;
  if ( (*(_DWORD *)(v2 + 20) & 0xFFFFFFF) != 3 )
    return 0;
  v3 = *(_QWORD *)(v2 - 24);
  v4 = *(_QWORD *)(v2 - 48);
  if ( a2 == v4 || a2 == v3 || v4 == v3 )
    return 0;
  if ( sub_157F0B0(v3) && v4 == sub_157F1C0(v3) )
    goto LABEL_25;
  if ( sub_157F0B0(v4) && v3 == sub_157F1C0(v4) )
    goto LABEL_22;
  if ( sub_157F0B0(v3) )
  {
    if ( sub_157F0B0(v4) )
    {
      if ( sub_157F1C0(v4) )
      {
        if ( a2 != sub_157F1C0(v4) )
        {
          v7 = sub_157F1C0(v4);
          if ( v7 == sub_157F1C0(v3) )
          {
            v8 = *(_QWORD *)(v4 + 48);
            v5 = v4 + 40;
            if ( v4 + 40 == v8 )
              goto LABEL_18;
            v9 = 0;
            do
            {
              v8 = *(_QWORD *)(v8 + 8);
              ++v9;
            }
            while ( v5 != v8 );
            if ( v9 != 1 )
            {
LABEL_18:
              v10 = *(_QWORD *)(v3 + 48);
              v5 = v3 + 40;
              if ( v3 + 40 != v10 )
              {
                v11 = 0;
                do
                {
                  v10 = *(_QWORD *)(v10 + 8);
                  ++v11;
                }
                while ( v5 != v10 );
                if ( v11 == 1 )
                {
LABEL_22:
                  v12 = a2;
                  v13 = v4;
                  return sub_1A682D0(a1, v13, v12, v5, v6);
                }
              }
              return 0;
            }
LABEL_25:
            v12 = a2;
            v13 = v3;
            return sub_1A682D0(a1, v13, v12, v5, v6);
          }
        }
      }
    }
  }
  return 0;
}
