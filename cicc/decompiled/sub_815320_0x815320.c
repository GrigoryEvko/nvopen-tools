// Function: sub_815320
// Address: 0x815320
//
unsigned __int8 *__fastcall sub_815320(__int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r13d
  int v7; // r12d
  const char *v8; // r12
  char v10; // al
  size_t v11; // rax
  char v12; // r13
  char v13; // r13
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rdx
  char *v17; // rax
  unsigned int v18; // r14d
  unsigned int v19; // [rsp+Ch] [rbp-74h] BYREF
  __int64 v20[4]; // [rsp+10h] [rbp-70h] BYREF
  char v21; // [rsp+30h] [rbp-50h]
  __int64 v22; // [rsp+38h] [rbp-48h]
  __int64 v23; // [rsp+40h] [rbp-40h]
  _BOOL4 v24; // [rsp+48h] [rbp-38h]
  char v25; // [rsp+4Ch] [rbp-34h]
  __int64 v26; // [rsp+50h] [rbp-30h]

  v6 = (int)a2;
  v7 = a3;
  v19 = 0;
  if ( (_DWORD)a3 && sub_736A10(a1) )
  {
    if ( (*(_BYTE *)(a1 + 89) & 0x28) == 8 && (*(_BYTE *)(a1 + 91) & 1) != 0 )
    {
LABEL_3:
      v8 = *(const char **)(a1 + 8);
      if ( v6 )
      {
        sub_809110(a1, a2, a3, a4, a5, a6);
        sub_823800(qword_4F18BE0);
        v11 = strlen(v8);
        sub_8238B0(qword_4F18BE0, v8, v11 + 1);
        v8 = *(const char **)(qword_4F18BE0 + 32);
        v12 = *(_BYTE *)(a1 + 205) >> 2;
        *(_BYTE *)(a1 + 205) = *(_BYTE *)(a1 + 205) & 0xE3 | 4;
        v13 = v12 & 7;
        if ( (*(_BYTE *)(a1 + 89) & 0x10) != 0 )
        {
          v8[strlen(v8) - 9] = 49;
        }
        else
        {
          v16 = *(_QWORD *)(a1 + 184);
          v17 = (char *)&v8[v16 + 1];
          if ( *v17 == 73 )
            v17 = (char *)&v8[v16 + 2];
          *v17 = 49;
        }
        *(_BYTE *)(a1 + 205) = *(_BYTE *)(a1 + 205) & 0xE3 | (4 * (v13 & 7));
        v14 = qword_4F18BE8;
        v15 = *(_QWORD *)qword_4F18BF0;
        qword_4F18BE8 = qword_4F18BF0;
        *(_QWORD *)qword_4F18BF0 = v14;
        qword_4F18BF0 = v15;
        if ( v15 )
        {
          qword_4F18BE0 = *(_QWORD *)(v15 + 8);
          return (unsigned __int8 *)v8;
        }
        qword_4F18BE0 = 0;
      }
      return (unsigned __int8 *)v8;
    }
  }
  else if ( (*(_BYTE *)(a1 + 89) & 0x28) == 8 )
  {
    goto LABEL_3;
  }
  a2 = &v19;
  if ( !sub_80A070(a1, &v19) )
    goto LABEL_3;
  v10 = *(_BYTE *)(a1 + 195);
  memset(v20, 0, sizeof(v20));
  v24 = (v10 & 8) != 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v25 = 0;
  v26 = 0;
  sub_809110(a1, &v19, a3, a4, a5, a6);
  sub_823800(qword_4F18BE0);
  v20[0] += 2;
  sub_8238B0(qword_4F18BE0, &unk_3C1BC40, 2);
  if ( v7 )
  {
    v18 = v19;
    if ( (*(_BYTE *)(a1 + 91) & 1) != 0 || sub_736A10(a1) )
      sub_80BD00((_QWORD *)a1, (__int64)v20);
    sub_8111C0(a1, v18, 0, v6, 0, 0, (__int64)v20);
  }
  else
  {
    sub_8111C0(a1, v19, 0, v6, 0, 0, (__int64)v20);
  }
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 174) - 1) <= 1u )
  {
    if ( (unsigned __int8)(((*(_BYTE *)(a1 + 205) >> 2) & 7) - 1) > 3u )
      sub_721090();
    v25 = a1209[(unsigned __int8)(((*(_BYTE *)(a1 + 205) >> 2) & 7) - 1)];
  }
  return sub_80B290(0, 1, (__int64)v20);
}
