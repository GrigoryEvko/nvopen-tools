// Function: sub_C3B3E0
// Address: 0xc3b3e0
//
__int64 __fastcall sub_C3B3E0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  char *v6; // rax
  char v7; // al
  char v8; // al
  __int64 result; // rax
  int v10; // r14d
  unsigned __int8 v11; // dl
  char v12; // dl
  __int64 v13; // rcx
  _QWORD v14[10]; // [rsp+0h] [rbp-50h] BYREF

  v6 = (char *)sub_C94E20(qword_4F863F0);
  if ( v6 )
    v7 = *v6;
  else
    v7 = qword_4F863F0[2];
  if ( v7 && *(_DWORD **)a1 == dword_3F657C0 )
    return sub_C3B200((__int64 *)a1, (__int64 *)a2, (__int64 *)a3, a4);
  v8 = *(_BYTE *)(a1 + 20) & 0xF7 | (*(_BYTE *)(a1 + 20) ^ *(_BYTE *)(a2 + 20)) & 8;
  *(_BYTE *)(a1 + 20) = v8;
  if ( (v8 & 6) != 0
    && (v8 & 7) != 3
    && (*(_BYTE *)(a2 + 20) & 7) != 3
    && (*(_BYTE *)(a2 + 20) & 6) != 0
    && (*(_BYTE *)(a3 + 20) & 6) != 0 )
  {
    sub_C33EB0(v14, (__int64 *)a3);
    v10 = sub_C39C10(a1, a2, (__int64)v14, 0);
    sub_C338F0((__int64)v14);
    result = sub_C36450(a1, a4, v10);
    if ( v10 )
      result = (unsigned int)result | 0x10;
    v11 = *(_BYTE *)(a1 + 20);
    if ( (v11 & 7) == 3 && (result & 8) == 0 && ((v11 ^ *(_BYTE *)(a3 + 20)) & 8) != 0 )
    {
      v12 = (8 * (a4 == 3)) | v11 & 0xF7;
      v13 = *(_QWORD *)a1;
      *(_BYTE *)(a1 + 20) = v12;
      if ( *(_DWORD *)(v13 + 20) == 2 )
        *(_BYTE *)(a1 + 20) = v12 & 0xF7;
    }
  }
  else
  {
    result = sub_C392E0((_BYTE *)a1, (_BYTE *)a2);
    if ( !(_DWORD)result )
      return sub_C3AC70(a1, a3, a4, 0);
  }
  return result;
}
