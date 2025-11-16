// Function: sub_29D81B0
// Address: 0x29d81b0
//
__int64 __fastcall sub_29D81B0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdi
  char v7; // al
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 result; // rax
  char v12; // cl
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // rdx
  __int64 v15; // r14
  __int64 v16; // rdi
  unsigned __int8 v17; // al
  unsigned __int8 v18; // cl
  int v19; // eax
  int v20; // edx
  __int64 v21; // r14
  int v22; // r13d
  __int64 v23; // [rsp+8h] [rbp-38h]

  while ( 2 )
  {
    v6 = *a1;
    v7 = *(_BYTE *)(a3 + 8);
    if ( *(_BYTE *)(a2 + 8) == 14 )
    {
      if ( v7 == 14 )
      {
        v8 = sub_B2BEC0(v6);
        v9 = a2;
        if ( !(*(_DWORD *)(a2 + 8) >> 8) )
          v9 = sub_AE4450(v8, a2);
        v23 = a2;
        a2 = v9;
        v10 = a3;
        if ( !(*(_DWORD *)(a3 + 8) >> 8) )
          goto LABEL_11;
      }
      else
      {
        v16 = sub_B2BEC0(v6);
        if ( *(_DWORD *)(a2 + 8) >> 8 )
        {
          v23 = a2;
          v10 = 0;
        }
        else
        {
          v10 = 0;
          v23 = a2;
          a2 = sub_AE4450(v16, a2);
        }
      }
    }
    else
    {
      if ( v7 == 14 )
      {
        v10 = a3;
        v23 = 0;
        v8 = sub_B2BEC0(v6);
        if ( *(_DWORD *)(a3 + 8) >> 8 )
          goto LABEL_6;
LABEL_11:
        a3 = sub_AE4450(v8, a3);
        goto LABEL_6;
      }
      sub_B2BEC0(v6);
      v10 = 0;
      v23 = 0;
    }
LABEL_6:
    if ( a3 == a2 )
      return 0;
    result = sub_29D7CF0((__int64)a1, *(unsigned __int8 *)(a2 + 8), *(unsigned __int8 *)(a3 + 8));
    if ( (_DWORD)result )
      return result;
    v12 = *(_BYTE *)(a2 + 8);
    switch ( v12 )
    {
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
      case 9:
      case 11:
        return result;
      case 12:
        v14 = *(_DWORD *)(a3 + 8) >> 8;
        v13 = *(_DWORD *)(a2 + 8) >> 8;
        return sub_29D7CF0((__int64)a1, v13, v14);
      case 13:
        v19 = *(_DWORD *)(a2 + 12);
        v20 = *(_DWORD *)(a3 + 12);
        if ( v19 != v20 )
        {
          v14 = (unsigned int)(v20 - 1);
          v13 = (unsigned int)(v19 - 1);
          return sub_29D7CF0((__int64)a1, v13, v14);
        }
        LOBYTE(v13) = *(_DWORD *)(a2 + 8) >> 8 != 0;
        LOBYTE(v14) = *(_DWORD *)(a3 + 8) >> 8 != 0;
        if ( (_BYTE)v14 != (_BYTE)v13 )
        {
          v14 = (unsigned __int8)v14;
          v13 = (unsigned __int8)v13;
          return sub_29D7CF0((__int64)a1, v13, v14);
        }
        result = sub_29D81B0(a1, **(_QWORD **)(a2 + 16), **(_QWORD **)(a3 + 16));
        if ( (_DWORD)result )
          return result;
        LODWORD(v21) = 0;
        v22 = *(_DWORD *)(a2 + 12) - 1;
        while ( (_DWORD)v21 != v22 )
        {
          v21 = (unsigned int)(v21 + 1);
          result = sub_29D81B0(
                     a1,
                     *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8 * v21),
                     *(_QWORD *)(*(_QWORD *)(a3 + 16) + 8 * v21));
          if ( (_DWORD)result )
            return result;
        }
        return 0;
      case 14:
        v14 = *(_DWORD *)(v10 + 8) >> 8;
        v13 = *(_DWORD *)(v23 + 8) >> 8;
        return sub_29D7CF0((__int64)a1, v13, v14);
      case 15:
        v13 = *(unsigned int *)(a2 + 12);
        v14 = *(unsigned int *)(a3 + 12);
        if ( (_DWORD)v13 != (_DWORD)v14 )
          return sub_29D7CF0((__int64)a1, v13, v14);
        if ( ((*(_DWORD *)(a2 + 8) & 0x200) != 0) != ((*(_DWORD *)(a3 + 8) & 0x200) != 0) )
        {
          v14 = (*(_DWORD *)(a3 + 8) & 0x200) != 0;
          v13 = (*(_DWORD *)(a2 + 8) & 0x200) != 0;
          return sub_29D7CF0((__int64)a1, v13, v14);
        }
        if ( !(_DWORD)v13 )
          return 0;
        v15 = 0;
        do
        {
          result = sub_29D81B0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 16) + v15), *(_QWORD *)(*(_QWORD *)(a3 + 16) + v15));
          if ( (_DWORD)result )
            return result;
          v15 += 8;
        }
        while ( v15 != 8 * v13 );
        return 0;
      case 16:
        v13 = *(_QWORD *)(a2 + 32);
        v14 = *(_QWORD *)(a3 + 32);
        if ( v14 == v13 )
          goto LABEL_28;
        return sub_29D7CF0((__int64)a1, v13, v14);
      case 17:
      case 18:
        v13 = *(unsigned int *)(a2 + 32);
        v14 = *(unsigned int *)(a3 + 32);
        v17 = v12 == 18;
        v18 = *(_BYTE *)(a3 + 8) == 18;
        if ( v18 == v17 )
        {
          if ( (_DWORD)v14 != (_DWORD)v13 )
            return sub_29D7CF0((__int64)a1, v13, v14);
LABEL_28:
          a3 = *(_QWORD *)(a3 + 24);
          a2 = *(_QWORD *)(a2 + 24);
          continue;
        }
        v14 = v18;
        v13 = v17;
        return sub_29D7CF0((__int64)a1, v13, v14);
      default:
        BUG();
    }
  }
}
