// Function: sub_8134A0
// Address: 0x8134a0
//
__int64 __fastcall sub_8134A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r13d
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 result; // rax
  __int64 v13; // [rsp+0h] [rbp-70h] BYREF
  __int64 v14; // [rsp+18h] [rbp-58h]
  char v15; // [rsp+20h] [rbp-50h]
  __int64 v16; // [rsp+28h] [rbp-48h]
  __int64 v17; // [rsp+30h] [rbp-40h]
  int v18; // [rsp+38h] [rbp-38h]
  char v19; // [rsp+3Ch] [rbp-34h]
  __int64 v20; // [rsp+40h] [rbp-30h]

  v6 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  sub_809110(a1, a2, a3, a4, a5, a6, 0, 0, 0);
  sub_823800(qword_4F18BE0);
  while ( 1 )
  {
    v13 += 2;
    sub_8238B0(qword_4F18BE0, &unk_3C1BC40, 2);
    sub_812C80(a1, 2u, 0, &v13);
    v7 = *(_QWORD *)(a1 + 40);
    if ( v7
      && *(_BYTE *)(v7 + 28) == 16
      && (*(_BYTE *)(a1 + 89) & 1) != 0
      && (*(_BYTE *)(*(_QWORD *)(a1 + 128) + 89LL) & 4) == 0 )
    {
      sub_80C040((__int64 *)a1, &v13);
    }
    sub_80B290(a1, 1, (__int64)&v13);
    result = (unsigned int)v20;
    if ( !(_DWORD)v20 || v6 == 1 )
      break;
    v6 = 1;
    v14 = 0;
    v15 = 0;
    v16 = 0;
    v17 = 0;
    v18 = 0;
    v19 = 0;
    v20 = 0;
    sub_809110(a1, 1, v8, v9, v10, v11, 0, 0, 0);
    sub_823800(qword_4F18BE0);
    HIDWORD(v20) = 1;
  }
  return result;
}
