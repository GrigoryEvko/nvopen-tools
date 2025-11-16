// Function: sub_17CEE20
// Address: 0x17cee20
//
__int64 __fastcall sub_17CEE20(__int64 a1, int a2)
{
  unsigned int v2; // r12d
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r14
  __int64 v14; // rax
  _QWORD v15[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = sub_1560260((_QWORD *)(a1 + 56), -1, a2);
  if ( !(_BYTE)v2 )
  {
    switch ( a2 )
    {
      case 4:
      case 13:
        if ( *(char *)(a1 + 23) < 0 )
        {
          v8 = sub_1648A40(a1);
          v10 = v8 + v9;
          if ( *(char *)(a1 + 23) < 0 )
            v10 -= sub_1648A40(a1);
          if ( (unsigned int)(v10 >> 4) )
            return v2;
        }
        goto LABEL_8;
      case 14:
      case 36:
        if ( *(char *)(a1 + 23) < 0 )
        {
          v4 = sub_1648A40(a1);
          v6 = v4 + v5;
          if ( *(char *)(a1 + 23) < 0 )
            v6 -= sub_1648A40(a1);
          if ( (unsigned int)(v6 >> 4) )
            return v2;
        }
        goto LABEL_8;
      case 37:
        if ( *(char *)(a1 + 23) >= 0 )
          goto LABEL_8;
        v11 = sub_1648A40(a1);
        v13 = v11 + v12;
        v14 = *(char *)(a1 + 23) >= 0 ? 0LL : sub_1648A40(a1);
        if ( v14 == v13 )
          goto LABEL_8;
        break;
      default:
        goto LABEL_8;
    }
    while ( *(_DWORD *)(*(_QWORD *)v14 + 8LL) <= 1u )
    {
      v14 += 16;
      if ( v13 == v14 )
      {
LABEL_8:
        v7 = *(_QWORD *)(a1 - 24);
        if ( *(_BYTE *)(v7 + 16) )
          return v2;
        v15[0] = *(_QWORD *)(v7 + 112);
        return (unsigned int)sub_1560260(v15, -1, a2);
      }
    }
  }
  return v2;
}
