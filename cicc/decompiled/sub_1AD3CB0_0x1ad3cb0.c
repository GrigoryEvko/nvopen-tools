// Function: sub_1AD3CB0
// Address: 0x1ad3cb0
//
void __fastcall sub_1AD3CB0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // r14
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rax

  v2 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_QWORD *)(v2 + 48) || *(__int16 *)(v2 + 18) < 0 )
  {
    v3 = sub_1625790(v2, 10);
    if ( v3 )
    {
      if ( *(_DWORD *)(a2 + 16) )
      {
        v4 = *(_QWORD *)(a2 + 8);
        v5 = v4 + ((unsigned __int64)*(unsigned int *)(a2 + 24) << 6);
        if ( v4 != v5 )
        {
          while ( 1 )
          {
            v6 = *(_QWORD *)(v4 + 24);
            if ( v6 != -8 && v6 != -16 )
              break;
            v4 += 64;
            if ( v5 == v4 )
              return;
          }
          while ( v4 != v5 )
          {
            v7 = *(_QWORD *)(v4 + 56);
            if ( v7 && *(_BYTE *)(v7 + 16) > 0x17u )
            {
              if ( (*(_QWORD *)(v7 + 48) || *(__int16 *)(v7 + 18) < 0)
                && (v8 = sub_1625790(*(_QWORD *)(v4 + 56), 10)) != 0 )
              {
                v3 = sub_1631960(v8, v3);
                sub_1625C10(v7, 10, v3);
              }
              else if ( (unsigned __int8)sub_15F2ED0(v7) || (unsigned __int8)sub_15F3040(v7) )
              {
                sub_1625C10(v7, 10, v3);
              }
            }
            v4 += 64;
            if ( v4 == v5 )
              break;
            while ( 1 )
            {
              v9 = *(_QWORD *)(v4 + 24);
              if ( v9 != -16 && v9 != -8 )
                break;
              v4 += 64;
              if ( v5 == v4 )
                return;
            }
          }
        }
      }
    }
  }
}
