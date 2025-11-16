// Function: sub_14DCF80
// Address: 0x14dcf80
//
__int64 __fastcall sub_14DCF80(__int64 a1, unsigned int a2, __int64 *a3, __int64 a4, _BYTE *a5, __int64 a6)
{
  unsigned __int8 v8; // al
  __int64 result; // rax
  int v10; // ecx
  __int64 *v11; // r12
  __int64 v12; // r14
  _BOOL4 v13; // r15d
  int v14; // ebx
  int v15; // eax
  int v16; // ecx
  int v17; // edi
  __int64 v18; // r15
  int v19; // [rsp+8h] [rbp-48h]
  int v21; // [rsp+18h] [rbp-38h] BYREF
  char v22; // [rsp+1Ch] [rbp-34h]

  if ( a2 - 11 <= 0x11 )
    return sub_14D6F90(a2, (_QWORD *)*a3, a3[1], (__int64)a5);
  if ( a2 - 36 <= 0xC )
    return sub_14D7A60(a2, *a3, *(_QWORD *)a1, a5);
  v8 = *(_BYTE *)(a1 + 16);
  if ( v8 <= 0x17u )
  {
    if ( v8 != 5 )
    {
LABEL_5:
      switch ( a2 )
      {
        case '6':
          v18 = a3[a4 - 1];
          if ( *(_BYTE *)(v18 + 16) || !(unsigned __int8)sub_14D90D0(a1 | 4, v18) )
            goto LABEL_17;
          result = sub_14DA350(a1 | 4, v18, a3, a4 - 1, a6);
          break;
        case '7':
          result = sub_15A2DC0(*a3, a3[1], a3[2], 0);
          break;
        case ';':
          result = sub_15A37D0(*a3, a3[1], 0);
          break;
        case '<':
          result = sub_15A3890(*a3, a3[1], a3[2], 0);
          break;
        case '=':
          result = sub_15A3950(*a3, a3[1], a3[2], 0);
          break;
        default:
LABEL_17:
          result = 0;
          break;
      }
      return result;
    }
    if ( *(_WORD *)(a1 + 18) != 32 )
      return sub_15A47B0(a1, a3, a4, *(_QWORD *)a1, 0, 0);
  }
  else if ( v8 != 56 )
  {
    goto LABEL_5;
  }
  result = sub_14DBA90(a1, (__int64 **)a3, a4, (__int64)a5, a6);
  if ( !result )
  {
    v10 = a4 - 1;
    v11 = a3 + 1;
    v12 = *(v11 - 1);
    v19 = v10;
    v13 = (*(_BYTE *)(a1 + 17) & 2) != 0;
    v14 = *(_BYTE *)(a1 + 17) >> 1 >> 1;
    v15 = sub_16348C0(a1);
    if ( v14 )
    {
      v22 = 1;
      v16 = v19;
      v17 = v15;
      v21 = v14 - 1;
    }
    else
    {
      v22 = 0;
      v16 = v19;
      v17 = v15;
    }
    return sub_15A2E80(v17, v12, (_DWORD)v11, v16, v13, (unsigned int)&v21, 0);
  }
  return result;
}
