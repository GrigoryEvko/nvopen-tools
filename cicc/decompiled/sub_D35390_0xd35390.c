// Function: sub_D35390
// Address: 0xd35390
//
bool __fastcall sub_D35390(char *a1, char *a2, __int64 a3, __int64 a4, char a5)
{
  char v6; // dl
  __int64 v8; // r10
  unsigned __int8 v9; // al
  bool result; // al
  __int64 v11; // rcx
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // [rsp-10h] [rbp-10h]

  v6 = *a1;
  v8 = 0;
  if ( (unsigned __int8)*a1 > 0x1Cu && (unsigned __int8)(v6 - 61) <= 1u )
  {
    v9 = *a2;
    v8 = *((_QWORD *)a1 - 4);
    if ( (unsigned __int8)*a2 <= 0x1Cu )
      return 0;
  }
  else
  {
    v9 = *a2;
    if ( (unsigned __int8)*a2 <= 0x1Cu )
      return 0;
  }
  if ( v9 != 61 && v9 != 62 )
    return 0;
  v11 = *((_QWORD *)a2 - 4);
  if ( !v11 || !v8 )
    return 0;
  if ( v6 == 61 )
  {
    v12 = *((_QWORD *)a1 + 1);
    if ( v9 != 61 )
      goto LABEL_12;
  }
  else
  {
    v12 = *(_QWORD *)(*((_QWORD *)a1 - 8) + 8LL);
    if ( v9 != 61 )
    {
LABEL_12:
      v13 = *(_QWORD *)(*((_QWORD *)a2 - 8) + 8LL);
      goto LABEL_13;
    }
  }
  v13 = *((_QWORD *)a2 + 1);
LABEL_13:
  v14 = sub_D35010(v12, v8, v13, v11, a3, a4, 1, a5);
  result = 0;
  if ( BYTE4(v14) )
    return (_DWORD)v14 == 1;
  return result;
}
