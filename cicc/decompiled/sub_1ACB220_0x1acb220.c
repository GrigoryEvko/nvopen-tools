// Function: sub_1ACB220
// Address: 0x1acb220
//
__int64 __fastcall sub_1ACB220(__int64 *a1, __int64 a2, __int64 a3)
{
  char v6; // dl
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 result; // rax
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rdx
  __int64 v14; // r14
  __int64 v15; // rdi
  int v16; // eax
  int v17; // edx
  __int64 v18; // r14
  int v19; // r13d
  __int64 v20; // [rsp+8h] [rbp-38h]

  while ( 2 )
  {
    v6 = *(_BYTE *)(a3 + 8);
    v7 = *a1;
    if ( *(_BYTE *)(a2 + 8) != 15 )
    {
      if ( v6 != 15 )
      {
        v10 = 0;
        sub_1632FA0(*(_QWORD *)(v7 + 40));
        v20 = 0;
        goto LABEL_6;
      }
      v10 = a3;
      v20 = 0;
      v8 = sub_1632FA0(*(_QWORD *)(v7 + 40));
      if ( *(_DWORD *)(a3 + 8) >> 8 )
        goto LABEL_6;
      goto LABEL_11;
    }
    if ( v6 != 15 )
    {
      v15 = sub_1632FA0(*(_QWORD *)(v7 + 40));
      if ( *(_DWORD *)(a2 + 8) >> 8 )
      {
        v20 = a2;
        v10 = 0;
      }
      else
      {
        v10 = 0;
        v20 = a2;
        a2 = sub_15A9650(v15, a2);
      }
      goto LABEL_6;
    }
    v8 = sub_1632FA0(*(_QWORD *)(v7 + 40));
    v9 = a2;
    if ( !(*(_DWORD *)(a2 + 8) >> 8) )
      v9 = sub_15A9650(v8, a2);
    v20 = a2;
    a2 = v9;
    v10 = a3;
    if ( !(*(_DWORD *)(a3 + 8) >> 8) )
LABEL_11:
      a3 = sub_15A9650(v8, a3);
LABEL_6:
    if ( a3 != a2 )
    {
      result = sub_1ACA9E0((__int64)a1, *(unsigned __int8 *)(a2 + 8), *(unsigned __int8 *)(a3 + 8));
      if ( (_DWORD)result )
        return result;
      switch ( *(_BYTE *)(a2 + 8) )
      {
        case 0:
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
        case 0xA:
          return result;
        case 1:
        case 9:
        case 0xE:
        case 0x10:
          v12 = *(_QWORD *)(a2 + 32);
          v13 = *(_QWORD *)(a3 + 32);
          if ( v13 != v12 )
            return sub_1ACA9E0((__int64)a1, v12, v13);
          a3 = *(_QWORD *)(a3 + 24);
          a2 = *(_QWORD *)(a2 + 24);
          continue;
        case 0xB:
          v13 = *(_DWORD *)(a3 + 8) >> 8;
          v12 = *(_DWORD *)(a2 + 8) >> 8;
          return sub_1ACA9E0((__int64)a1, v12, v13);
        case 0xC:
          v16 = *(_DWORD *)(a2 + 12);
          v17 = *(_DWORD *)(a3 + 12);
          if ( v17 != v16 )
          {
            v13 = (unsigned int)(v17 - 1);
            v12 = (unsigned int)(v16 - 1);
            return sub_1ACA9E0((__int64)a1, v12, v13);
          }
          LOBYTE(v12) = *(_DWORD *)(a2 + 8) >> 8 != 0;
          LOBYTE(v13) = *(_DWORD *)(a3 + 8) >> 8 != 0;
          if ( (_BYTE)v13 != (_BYTE)v12 )
          {
            v13 = (unsigned __int8)v13;
            v12 = (unsigned __int8)v12;
            return sub_1ACA9E0((__int64)a1, v12, v13);
          }
          result = sub_1ACB220(a1, **(_QWORD **)(a2 + 16), **(_QWORD **)(a3 + 16));
          if ( (_DWORD)result )
            return result;
          LODWORD(v18) = 0;
          v19 = *(_DWORD *)(a2 + 12) - 1;
          break;
        case 0xD:
          v12 = *(unsigned int *)(a2 + 12);
          v13 = *(unsigned int *)(a3 + 12);
          if ( (_DWORD)v12 != (_DWORD)v13 )
            return sub_1ACA9E0((__int64)a1, v12, v13);
          if ( ((*(_DWORD *)(a2 + 8) & 0x200) != 0) != ((*(_DWORD *)(a3 + 8) & 0x200) != 0) )
          {
            v13 = (*(_DWORD *)(a3 + 8) & 0x200) != 0;
            v12 = (*(_DWORD *)(a2 + 8) & 0x200) != 0;
            return sub_1ACA9E0((__int64)a1, v12, v13);
          }
          if ( !(_DWORD)v12 )
            return 0;
          v14 = 0;
          do
          {
            result = sub_1ACB220(a1, *(_QWORD *)(*(_QWORD *)(a2 + 16) + v14), *(_QWORD *)(*(_QWORD *)(a3 + 16) + v14));
            if ( (_DWORD)result )
              return result;
            v14 += 8;
          }
          while ( 8 * v12 != v14 );
          return 0;
        case 0xF:
          v13 = *(_DWORD *)(v10 + 8) >> 8;
          v12 = *(_DWORD *)(v20 + 8) >> 8;
          return sub_1ACA9E0((__int64)a1, v12, v13);
      }
      while ( (_DWORD)v18 != v19 )
      {
        v18 = (unsigned int)(v18 + 1);
        result = sub_1ACB220(
                   a1,
                   *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8 * v18),
                   *(_QWORD *)(*(_QWORD *)(a3 + 16) + 8 * v18));
        if ( (_DWORD)result )
          return result;
      }
    }
    return 0;
  }
}
