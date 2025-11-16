// Function: sub_344A030
// Address: 0x344a030
//
bool __fastcall sub_344A030(_DWORD *a1, __int64 a2, __int64 a3, __int64 a4, bool a5)
{
  unsigned __int16 v6; // cx
  int v7; // eax
  __int64 v8; // rdi
  unsigned int v9; // r13d
  bool result; // al
  char v11; // r14
  __int64 v12; // rdi
  unsigned int v13; // r12d
  __int64 v14; // rdi
  unsigned int v15; // ebx
  _QWORD v16[8]; // [rsp+0h] [rbp-40h] BYREF

  if ( (_WORD)a3 == 2 )
  {
    v14 = *(_QWORD *)(a2 + 96);
    v15 = *(_DWORD *)(v14 + 32);
    if ( v15 <= 0x40 )
      return *(_QWORD *)(v14 + 24) == 1;
    else
      return v15 - 1 == (unsigned int)sub_C444A0(v14 + 24);
  }
  v16[0] = a3;
  v16[1] = a4;
  if ( !(_WORD)a3 )
  {
    v11 = sub_3007030((__int64)v16);
    if ( sub_30070B0((__int64)v16) )
      goto LABEL_26;
    if ( !v11 )
      goto LABEL_6;
LABEL_14:
    v7 = a1[16];
    if ( v7 == 1 )
      goto LABEL_8;
    goto LABEL_15;
  }
  v6 = a3 - 17;
  if ( (unsigned __int16)(a3 - 10) <= 6u || (unsigned __int16)(a3 - 126) <= 0x31u )
  {
    if ( v6 <= 0xD3u )
      goto LABEL_26;
    goto LABEL_14;
  }
  if ( v6 > 0xD3u )
  {
LABEL_6:
    v7 = a1[15];
    goto LABEL_7;
  }
LABEL_26:
  v7 = a1[17];
LABEL_7:
  if ( v7 == 1 )
  {
LABEL_8:
    v8 = *(_QWORD *)(a2 + 96);
    v9 = *(_DWORD *)(v8 + 32);
    if ( v9 <= 0x40 )
      result = *(_QWORD *)(v8 + 24) == 1;
    else
      result = v9 - 1 == (unsigned int)sub_C444A0(v8 + 24);
    if ( a5 )
      return **(_WORD **)(a2 + 48) != 2;
    return result;
  }
LABEL_15:
  if ( (v7 & 0xFFFFFFFD) != 0 )
    BUG();
  v12 = *(_QWORD *)(a2 + 96);
  result = a5;
  v13 = *(_DWORD *)(v12 + 32);
  if ( v13 )
  {
    result = v13 <= 0x40
           ? 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v13) == *(_QWORD *)(v12 + 24)
           : v13 == (unsigned int)sub_C445E0(v12 + 24);
    if ( result )
      return a5;
  }
  return result;
}
