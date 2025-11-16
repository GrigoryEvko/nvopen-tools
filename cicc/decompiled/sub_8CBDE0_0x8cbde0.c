// Function: sub_8CBDE0
// Address: 0x8cbde0
//
__int64 __fastcall sub_8CBDE0(__int64 *a1)
{
  __int64 v2; // r13
  __int64 result; // rax
  __int64 v4; // rax
  _QWORD *v5; // r13
  __int64 v6; // rdi
  _QWORD *v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  char v10; // al
  char v11; // al
  __int64 v12; // rdi

  v2 = *a1;
  if ( !a1[1] )
    return (__int64)sub_8C7090(7, (__int64)a1);
  if ( !v2 )
    return (__int64)sub_8C7090(7, (__int64)a1);
  if ( !(unsigned int)sub_8C6B40(*a1) )
    return (__int64)sub_8C7090(7, (__int64)a1);
  v4 = sub_8C84B0(v2, *(_QWORD *)(*(_QWORD *)v2 + 32LL));
  if ( !v4 )
  {
    v8 = sub_8C6F20(v2);
    v4 = sub_8C84B0(v2, v8);
    if ( !v4 )
      return (__int64)sub_8C7090(7, (__int64)a1);
  }
  v5 = *(_QWORD **)(v4 + 88);
  sub_8CBB20(7u, (__int64)a1, v5);
  v6 = a1[15];
  v7 = *(_QWORD **)(v6 + 32);
  if ( (!v7 || v6 == *v7 && v6 != v7[1])
    && !*(_QWORD *)(v6 + 8)
    && (v9 = v5[15], !*(_QWORD *)(v9 + 8))
    && ((v10 = *(_BYTE *)(v6 + 140), (unsigned __int8)(v10 - 9) <= 2u) || v10 == 2 && (*(_BYTE *)(v6 + 161) & 8) != 0)
    && ((v11 = *(_BYTE *)(v9 + 140), (unsigned __int8)(v11 - 9) <= 2u) || v11 == 2 && (*(_BYTE *)(v9 + 161) & 8) != 0) )
  {
    sub_8CBB20(6u, a1[15], (_QWORD *)v9);
    v12 = a1[15];
    result = *(unsigned __int8 *)(v12 + 140);
    if ( (_BYTE)result == *(_BYTE *)(v5[15] + 140LL) )
    {
      if ( (unsigned __int8)(result - 9) > 2u )
        return sub_8CA420(v12);
      else
        return sub_8CAE10(v12);
    }
  }
  else
  {
    result = (__int64)&dword_4F077C4;
    if ( dword_4F077C4 != 2 )
      return sub_8DED30(v6, v5[15], 261);
  }
  return result;
}
