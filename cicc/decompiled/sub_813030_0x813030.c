// Function: sub_813030
// Address: 0x813030
//
unsigned __int8 *__fastcall sub_813030(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  _BOOL4 v4; // ebx
  __int64 v5; // r8
  __int64 v6; // r9
  char v8; // al
  _QWORD v9[4]; // [rsp+0h] [rbp-70h] BYREF
  char v10; // [rsp+20h] [rbp-50h]
  __int64 v11; // [rsp+28h] [rbp-48h]
  __int64 v12; // [rsp+30h] [rbp-40h]
  _BOOL4 v13; // [rsp+38h] [rbp-38h]
  char v14; // [rsp+3Ch] [rbp-34h]
  __int64 v15; // [rsp+40h] [rbp-30h]

  v4 = sub_736A30(a1);
  if ( (*(_BYTE *)(a1 + 89) & 0x28) == 8 && (!v4 || (*(_BYTE *)(a1 + 91) & 1) != 0) )
    return *(unsigned __int8 **)(a1 + 8);
  v8 = *(_BYTE *)(a1 + 170);
  v9[3] = 0;
  v13 = (v8 & 0x20) != 0;
  v10 = 0;
  v11 = 0;
  v12 = 0;
  v14 = 0;
  v15 = 0;
  sub_809110(a1, a2, v2, v3, v5, v6, 0, 0, 0);
  sub_823800(qword_4F18BE0);
  v9[0] += 2LL;
  sub_8238B0(qword_4F18BE0, &unk_3C1BC40, 2);
  if ( v4 )
    sub_80BD00((_QWORD *)a1, (__int64)v9);
  sub_812EE0(a1, v9);
  return sub_80B290(0, 1, (__int64)v9);
}
