// Function: sub_397D230
// Address: 0x397d230
//
unsigned __int64 __fastcall sub_397D230(__int64 a1, __int64 a2, __int64 a3, const char *a4)
{
  unsigned __int64 result; // rax
  __int64 v7; // rax
  void *v8; // rdi
  char *v9; // rsi
  size_t v10; // r12
  size_t v11; // rdx
  char *v12; // r8
  __int64 v13; // rcx
  int v14; // r12d
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // [rsp-8h] [rbp-78h]
  _QWORD v19[2]; // [rsp+0h] [rbp-70h] BYREF
  char v20; // [rsp+10h] [rbp-60h] BYREF
  void *v21; // [rsp+20h] [rbp-50h] BYREF
  __int64 v22; // [rsp+28h] [rbp-48h]
  __int64 v23; // [rsp+30h] [rbp-40h]
  __int64 v24; // [rsp+38h] [rbp-38h]
  int v25; // [rsp+40h] [rbp-30h]
  _QWORD *v26; // [rsp+48h] [rbp-28h]

  if ( !strcmp(a4, "private") )
  {
    result = *(unsigned int *)(sub_1E0A0C0(*(_QWORD *)(a1 + 264)) + 16);
    switch ( result )
    {
      case 0uLL:
        return result;
      case 1uLL:
      case 3uLL:
        v11 = 2;
        v12 = ".L";
        goto LABEL_12;
      case 2uLL:
      case 4uLL:
        v11 = 1;
        v12 = "L";
        goto LABEL_12;
      case 5uLL:
        v11 = 1;
        v12 = "$";
LABEL_12:
        v13 = *(_QWORD *)(a3 + 24);
        v9 = v12;
        if ( v11 > *(_QWORD *)(a3 + 16) - v13 )
          return sub_16E7EE0(a3, v9, v11);
        LODWORD(result) = 0;
        do
        {
          v15 = (unsigned int)result;
          result = (unsigned int)(result + 1);
          *(_BYTE *)(v13 + v15) = v12[v15];
        }
        while ( (unsigned int)result < (unsigned int)v11 );
        *(_QWORD *)(a3 + 24) += v11;
        break;
    }
  }
  else if ( !strcmp(a4, "comment") )
  {
    v7 = *(_QWORD *)(a1 + 240);
    v8 = *(void **)(a3 + 24);
    v9 = *(char **)(v7 + 48);
    v10 = *(_QWORD *)(v7 + 56);
    result = *(_QWORD *)(a3 + 16) - (_QWORD)v8;
    if ( v10 > result )
    {
      v11 = v10;
      return sub_16E7EE0(a3, v9, v11);
    }
    else if ( v10 )
    {
      result = (unsigned __int64)memcpy(v8, v9, v10);
      *(_QWORD *)(a3 + 24) += v10;
    }
  }
  else
  {
    if ( strcmp(a4, "uid") )
    {
      v19[1] = 0;
      v19[0] = &v20;
      v20 = 0;
      v25 = 1;
      v21 = &unk_49EFBE0;
      v24 = 0;
      v23 = 0;
      v22 = 0;
      v26 = v19;
      v16 = sub_1263B40((__int64)&v21, "Unknown special formatter '");
      v17 = sub_1263B40(v16, a4);
      v18 = sub_1263B40(v17, "' for machine instr: ");
      sub_1E1A330(a2, v18, 1, 0, 0, 1, 0);
      if ( v24 != v22 )
        sub_16E7BA0((__int64 *)&v21);
      sub_16BD160((__int64)v26, 1u);
    }
    if ( *(_QWORD *)(a1 + 728) != a2 || (v14 = *(_DWORD *)(a1 + 736), v14 != (unsigned int)sub_396DD70(a1)) )
    {
      ++*(_DWORD *)(a1 + 740);
      *(_QWORD *)(a1 + 728) = a2;
      *(_DWORD *)(a1 + 736) = sub_396DD70(a1);
    }
    return sub_16E7A90(a3, *(unsigned int *)(a1 + 740));
  }
  return result;
}
