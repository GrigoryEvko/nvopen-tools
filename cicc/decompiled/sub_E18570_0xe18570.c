// Function: sub_E18570
// Address: 0xe18570
//
__int64 __fastcall sub_E18570(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v6; // rax
  char *v7; // rdx
  __int64 v8; // r12
  __int64 v10; // rdx
  char *v11; // rdx
  unsigned __int64 v12; // rax
  __int64 v13; // rcx
  int v14; // r13d
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rax
  __int64 v21[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = *(char **)a1;
  v7 = *(char **)(a1 + 8);
  if ( v7 == *(char **)a1 || *v6 != 83 )
    return 0;
  *(_QWORD *)a1 = v6 + 1;
  if ( v7 == v6 + 1 )
  {
LABEL_8:
    v21[0] = 0;
    if ( !(unsigned __int8)sub_E0E050((char **)a1, v21) )
    {
      v11 = *(char **)a1;
      v12 = ++v21[0];
      if ( v11 != *(char **)(a1 + 8) && *v11 == 95 )
      {
        v13 = *(_QWORD *)(a1 + 296);
        *(_QWORD *)a1 = v11 + 1;
        if ( v12 < (*(_QWORD *)(a1 + 304) - v13) >> 3 )
          return *(_QWORD *)(v13 + 8 * v12);
      }
    }
    return 0;
  }
  v10 = (unsigned __int8)v6[1];
  if ( (unsigned __int8)(v10 - 97) > 0x19u )
  {
    if ( (_BYTE)v10 == 95 )
    {
      *(_QWORD *)a1 = v6 + 2;
      v20 = *(_QWORD *)(a1 + 296);
      if ( v20 != *(_QWORD *)(a1 + 304) )
        return *(_QWORD *)v20;
      return 0;
    }
    goto LABEL_8;
  }
  switch ( (char)v10 )
  {
    case 'a':
      v14 = 0;
      break;
    case 'b':
      v14 = 1;
      break;
    case 'd':
      v14 = 5;
      break;
    case 'i':
      v14 = 3;
      break;
    case 'o':
      v14 = 4;
      break;
    case 's':
      v14 = 2;
      break;
    default:
      return 0;
  }
  *(_QWORD *)a1 = v6 + 2;
  v15 = sub_E0E790(a1 + 816, 16, v10, (unsigned __int8)(v10 - 97), a5, a6);
  v8 = v15;
  if ( v15 )
  {
    *(_DWORD *)(v15 + 12) = v14;
    *(_WORD *)(v15 + 8) = 16432;
    *(_BYTE *)(v15 + 10) = *(_BYTE *)(v15 + 10) & 0xF0 | 5;
    *(_QWORD *)v15 = &unk_49DFFC8;
    v21[0] = sub_E0F930((_QWORD *)a1, v15);
    if ( v21[0] != v8 )
    {
      sub_E18380(a1 + 296, v21, v16, v17, v18, v19);
      return v21[0];
    }
  }
  return v8;
}
