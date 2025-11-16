// Function: sub_3074BE0
// Address: 0x3074be0
//
__int64 __fastcall sub_3074BE0(__int64 a1, __int64 a2)
{
  char v2; // al
  char v4; // al
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  int v11; // r13d
  __int64 v12; // rax
  unsigned __int8 *v13; // rdi

  v2 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 > 0x1Cu )
  {
    if ( v2 != 60 )
    {
      if ( v2 == 61 )
      {
        v10 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
        if ( (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17 <= 1 )
          v10 = **(_QWORD **)(v10 + 16);
        v11 = *(_DWORD *)(v10 + 8) >> 8;
        if ( v11 == 4 && (_BYTE)qword_502CFE8 )
          return 1;
        if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL) + 1280LL) == 1 )
        {
          if ( (_BYTE)qword_502CF08 )
          {
            v12 = sub_B43CB0(a2);
            if ( (unsigned __int8)sub_CE9220(v12) )
            {
              if ( v11 == 101 )
                return 1;
              v13 = sub_98ACB0(*(unsigned __int8 **)(a2 - 32), 6u);
              if ( *v13 == 22 )
              {
                if ( (unsigned __int8)sub_B2D680((__int64)v13) )
                  return 1;
              }
            }
          }
        }
      }
      else if ( v2 == 77 && *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL) + 1280LL) == 1 )
      {
        if ( (_BYTE)qword_502CF08 )
        {
          v5 = sub_B43CB0(a2);
          if ( (unsigned __int8)sub_CE9220(v5) )
          {
            if ( (unsigned __int8)sub_3071B90(*(unsigned __int8 **)(a2 - 32), a2, v6, v7, v8, v9) )
              return 1;
          }
        }
      }
      return 0xFFFFFFFFLL;
    }
    return 5;
  }
  if ( v2 != 22 )
    return 0xFFFFFFFFLL;
  v4 = sub_CE9220(*(_QWORD *)(a2 + 24));
  if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL) + 1280LL) != 1 || !(_BYTE)qword_502CF08 )
  {
    if ( v4 )
      return 0xFFFFFFFFLL;
    goto LABEL_9;
  }
  if ( !v4 )
  {
LABEL_9:
    if ( !(unsigned __int8)sub_B2D680(a2) )
      return 0xFFFFFFFFLL;
    return 5;
  }
  if ( (unsigned __int8)sub_B2D680(a2) )
    return 0xFFFFFFFFLL;
  return 1;
}
