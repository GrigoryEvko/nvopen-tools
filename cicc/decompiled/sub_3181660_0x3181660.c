// Function: sub_3181660
// Address: 0x3181660
//
char __fastcall sub_3181660(int a1, __int64 *a2, __int64 a3, __int64 *a4)
{
  unsigned __int8 *v6; // r12
  char result; // al
  char v8; // al
  unsigned int v9; // edi
  int v10; // eax
  unsigned int v11; // eax
  __int64 v12; // rdi
  int v13; // eax
  int v14; // edi
  unsigned __int8 v15; // al
  __int64 v16; // r8
  __int64 v17; // r8
  int v18; // edi
  unsigned __int8 v19; // al
  __int64 v20; // r8

  if ( a2 == (__int64 *)a3 )
    return 1;
  v6 = (unsigned __int8 *)a2;
  switch ( a1 )
  {
    case 0:
      v10 = sub_3108990((__int64)a2);
      if ( v10 <= 8 )
      {
        if ( v10 > 6 )
          return 0;
        return sub_3181380(a2, a3, a4, v10);
      }
      if ( v10 != 24 )
        return sub_3181380(a2, a3, a4, v10);
      return 0;
    case 1:
      return (unsigned int)sub_3108990((__int64)a2) - 7 <= 1;
    case 2:
      v11 = sub_3108990((__int64)a2);
      if ( v11 == 8 )
        return 1;
      if ( v11 == 24 || v11 == 7 )
        return 0;
      return sub_3181170((unsigned __int8 *)a2, a3, a4, v11);
    case 3:
      result = 0;
      if ( *(_BYTE *)a2 != 85 )
        return result;
      v12 = *(a2 - 4);
      if ( !v12 || *(_BYTE *)v12 || *(_QWORD *)(v12 + 24) != a2[10] )
        return result;
      v13 = sub_3108960(v12);
      if ( v13 > 1 )
        return (unsigned int)(v13 - 7) <= 1;
      if ( v13 < 0 )
        return 0;
      do
      {
        v14 = 23;
        v6 = sub_BD3990(*(unsigned __int8 **)&v6[-32 * (*((_DWORD *)v6 + 1) & 0x7FFFFFF)], (__int64)a2);
        v15 = *v6;
        if ( *v6 > 0x1Cu )
        {
          if ( v15 == 85 )
          {
            v16 = *((_QWORD *)v6 - 4);
            v14 = 21;
            if ( v16 && !*(_BYTE *)v16 && *(_QWORD *)(v16 + 24) == *((_QWORD *)v6 + 10) )
              v14 = sub_3108960(*((_QWORD *)v6 - 4));
          }
          else
          {
            v14 = 2 * (v15 != 34) + 21;
          }
        }
      }
      while ( (unsigned __int8)sub_3108CA0(v14) );
      goto LABEL_37;
    case 4:
      v8 = *(_BYTE *)a2;
      if ( *(_BYTE *)a2 <= 0x1Cu )
      {
        v9 = 23;
        return sub_3108DC0(v9);
      }
      if ( v8 != 85 )
      {
        v9 = 2 * (v8 != 34) + 21;
        return sub_3108DC0(v9);
      }
      v17 = *(a2 - 4);
      v9 = 21;
      if ( !v17 )
        return sub_3108DC0(v9);
      if ( *(_BYTE *)v17 )
        return sub_3108DC0(v9);
      if ( *(_QWORD *)(v17 + 24) != a2[10] )
        return sub_3108DC0(v9);
      v9 = sub_3108960(*(a2 - 4));
      if ( v9 > 1 )
        return sub_3108DC0(v9);
      do
      {
        v18 = 23;
        v6 = sub_BD3990(*(unsigned __int8 **)&v6[-32 * (*((_DWORD *)v6 + 1) & 0x7FFFFFF)], (__int64)a2);
        v19 = *v6;
        if ( *v6 > 0x1Cu )
        {
          if ( v19 == 85 )
          {
            v20 = *((_QWORD *)v6 - 4);
            v18 = 21;
            if ( v20 && !*(_BYTE *)v20 && *(_QWORD *)(v20 + 24) == *((_QWORD *)v6 + 10) )
              v18 = sub_3108960(*((_QWORD *)v6 - 4));
          }
          else
          {
            v18 = 2 * (v19 != 34) + 21;
          }
        }
      }
      while ( (unsigned __int8)sub_3108CA0(v18) );
LABEL_37:
      result = a3 == (_QWORD)v6;
      break;
    default:
      BUG();
  }
  return result;
}
