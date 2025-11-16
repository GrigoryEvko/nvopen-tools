// Function: sub_80CF50
// Address: 0x80cf50
//
__int64 __fastcall sub_80CF50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v7; // r12
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r14
  char *v11; // r15
  __int64 *v12; // rdx
  __int64 v13; // rcx
  __int64 result; // rax
  __int64 v15; // rax
  size_t v16; // rax
  _QWORD s[8]; // [rsp+0h] [rbp-C0h] BYREF
  _QWORD v18[4]; // [rsp+40h] [rbp-80h] BYREF
  char v19; // [rsp+60h] [rbp-60h]
  __int64 v20; // [rsp+68h] [rbp-58h]
  __int64 v21; // [rsp+70h] [rbp-50h]
  int v22; // [rsp+78h] [rbp-48h]
  char v23; // [rsp+7Ch] [rbp-44h]
  __int64 v24; // [rsp+80h] [rbp-40h]

  memset(v18, 0, sizeof(v18));
  v7 = *(char **)(a1 + 8);
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  sub_809110(a1, a2, a3, a4, a5, a6, s[0], s[1], s[2]);
  sub_823800(qword_4F18BE0);
  v18[0] += 2LL;
  sub_8238B0(qword_4F18BE0, &unk_3C1BC40, 2);
  sub_80BD00((_QWORD *)a1, (__int64)v18);
  if ( !v7 )
  {
    if ( !(_DWORD)a2
      || ((*(_BYTE *)(a1 + 89) & 8) != 0 ? (v7 = *(char **)(a1 + 24)) : (v7 = *(char **)(a1 + 8)),
          !v7 && ((*(_BYTE *)(a1 + 172) & 2) == 0 || (v7 = (char *)sub_808FB0(*(_QWORD *)(a1 + 120), s)) == 0)) )
    {
      v7 = (char *)s;
      v15 = sub_737880(a1);
      snprintf((char *)s, 0x32u, "%llu", v15);
    }
  }
  if ( *v7 == 95 && v7[1] == 90 )
  {
    v16 = strlen(v7 + 2);
    v18[0] += v16;
    sub_8238B0(qword_4F18BE0, v7 + 2, v16);
  }
  else
  {
    sub_80BC40(v7, v18);
  }
  v8 = qword_4F18BE0;
  ++v18[0];
  v9 = *(_QWORD *)(qword_4F18BE0 + 16);
  if ( (unsigned __int64)(v9 + 1) > *(_QWORD *)(qword_4F18BE0 + 8) )
  {
    sub_823810(qword_4F18BE0);
    v8 = qword_4F18BE0;
    v9 = *(_QWORD *)(qword_4F18BE0 + 16);
  }
  *(_BYTE *)(*(_QWORD *)(v8 + 32) + v9) = 0;
  v10 = *(_QWORD *)(v8 + 16);
  *(_QWORD *)(v8 + 16) = v10 + 1;
  v11 = (char *)sub_7E1510(v10 + 1);
  strcpy(v11, *(const char **)(qword_4F18BE0 + 32));
  if ( !(_DWORD)a2 && (unsigned __int8)(*(_BYTE *)(a1 + 174) - 1) <= 1u )
    *(_QWORD *)(a1 + 184) += v10 - strlen(v7);
  *(_BYTE *)(a1 + 91) |= 1u;
  v12 = (__int64 *)qword_4F18BF0;
  *(_QWORD *)(a1 + 8) = v11;
  v13 = qword_4F18BE8;
  result = *v12;
  qword_4F18BE8 = (__int64)v12;
  *v12 = v13;
  qword_4F18BF0 = result;
  if ( result )
    result = *(_QWORD *)(result + 8);
  qword_4F18BE0 = result;
  return result;
}
