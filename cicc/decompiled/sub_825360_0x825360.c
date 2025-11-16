// Function: sub_825360
// Address: 0x825360
//
__int64 __fastcall sub_825360(int a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  int v7; // ebx
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 result; // rax
  int v11; // edi
  __int64 v12; // rdx

  v6 = a4;
  v7 = a3;
  qword_4F5F768 = 0;
  v8 = sub_822B10(24, a2, a3, a4, a5, a6);
  v9 = qword_4F5F768;
  *(_QWORD *)v8 = v6;
  *(_QWORD *)(v8 + 16) = v9;
  *(_DWORD *)(v8 + 8) = v7;
  while ( 1 )
  {
    qword_4F5F768 = v9;
    sub_822B90(v8, 24);
    result = *(_BYTE *)(v6 + 193) & 0x10;
    if ( (*(_BYTE *)(v6 + 193) & 0x10) != 0 )
    {
      if ( *(_DWORD *)(v6 + 160) )
        goto LABEL_10;
      if ( (unsigned __int8)(*(_BYTE *)(v6 + 174) - 1) > 1u )
        goto LABEL_4;
      goto LABEL_29;
    }
    if ( (*(_QWORD *)(v6 + 200) & 0x8000001000000LL) == 0x8000000000000LL && (*(_BYTE *)(v6 + 192) & 2) == 0 )
    {
      if ( *(_DWORD *)(v6 + 160) )
      {
        result = 0x8000000000000LL;
        goto LABEL_26;
      }
      if ( (unsigned __int8)(*(_BYTE *)(v6 + 174) - 1) > 1u )
      {
LABEL_25:
        result = *(_QWORD *)(v6 + 200) & 0x8000001000000LL;
LABEL_26:
        if ( result == 0x8000000000000LL && (*(_BYTE *)(v6 + 192) & 2) == 0 )
          goto LABEL_9;
        goto LABEL_8;
      }
LABEL_29:
      if ( (*(_BYTE *)(v6 + 205) & 0x20) != 0 )
      {
        v12 = *(_QWORD *)(v6 + 320);
        if ( v12 )
        {
          v6 = *(_QWORD *)(v6 + 320);
          if ( (*(_BYTE *)(v12 + 193) & 0x10) == 0 )
          {
            result = *(_QWORD *)(v12 + 200) & 0x8000001000000LL;
            if ( result != 0x8000000000000LL || (*(_BYTE *)(v12 + 192) & 2) != 0 )
              goto LABEL_4;
          }
          goto LABEL_9;
        }
      }
      if ( (_BYTE)result )
        goto LABEL_4;
      goto LABEL_25;
    }
LABEL_8:
    if ( !v7 )
      goto LABEL_4;
LABEL_9:
    if ( !*(_DWORD *)(v6 + 160) )
      goto LABEL_4;
LABEL_10:
    if ( *(_BYTE *)(v6 + 174) == 6 || (*(_BYTE *)(v6 + 206) & 2) != 0 )
      goto LABEL_4;
    if ( (_DWORD)a2 )
    {
      result = *(unsigned __int8 *)(v6 + 198);
      if ( (result & 0x10) == 0 )
        break;
    }
    if ( a1 )
    {
      result = *(unsigned __int8 *)(v6 + 198);
      if ( (result & 8) == 0 )
      {
        if ( (_DWORD)a2 )
          *(_BYTE *)(v6 + 198) = result | 0x10;
        goto LABEL_21;
      }
    }
LABEL_4:
    v8 = qword_4F5F768;
    if ( !qword_4F5F768 )
      return result;
LABEL_5:
    v6 = *(_QWORD *)v8;
    v7 = *(_DWORD *)(v8 + 8);
    v9 = *(_QWORD *)(v8 + 16);
  }
  *(_BYTE *)(v6 + 198) = result | 0x10;
  if ( !a1 )
    goto LABEL_15;
LABEL_21:
  *(_BYTE *)(v6 + 198) |= 8u;
LABEL_15:
  v11 = *(_DWORD *)(v6 + 164);
  qword_4F5F760 = v6;
  result = sub_75AFC0(v11, (__int64 (__fastcall *)(_QWORD, _QWORD))sub_8252B0, 0, 0, 0, 0, 0);
  v8 = qword_4F5F768;
  if ( qword_4F5F768 )
    goto LABEL_5;
  return result;
}
