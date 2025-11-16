// Function: sub_E0D140
// Address: 0xe0d140
//
__int64 __fastcall sub_E0D140(__int64 a1, __int64 *a2)
{
  __int64 v2; // rdx
  _BYTE *v3; // rax
  _BYTE *v5; // rax
  __int64 v6; // r14
  __int64 v7; // r13
  bool v8; // zf
  const char *v9; // r13
  size_t v10; // rax
  __int64 v11; // [rsp+8h] [rbp-38h] BYREF
  size_t v12; // [rsp+10h] [rbp-30h] BYREF
  const char *v13; // [rsp+18h] [rbp-28h]

  v2 = *a2;
  if ( !*a2 )
    goto LABEL_5;
  v3 = (_BYTE *)a2[1];
  if ( *v3 == 81 )
  {
    v5 = &v3[-*(_QWORD *)(a1 + 8)];
    v6 = *(int *)(a1 + 16);
    if ( (__int64)v5 >= v6 )
      goto LABEL_8;
    *(_DWORD *)(a1 + 16) = (_DWORD)v5;
    v7 = a2[1];
    v8 = (*a2)-- == 1;
    v12 = 0;
    v13 = 0;
    a2[1] = v7 + 1;
    if ( v8 )
      goto LABEL_8;
    if ( !(unsigned __int8)sub_E0CFB0(a2, &v11) )
      goto LABEL_8;
    if ( v7 - *(_QWORD *)(a1 + 8) < v11 )
      goto LABEL_8;
    v9 = (const char *)(v7 - v11);
    v10 = strlen(v9);
    v13 = v9;
    v12 = v10;
    if ( !v10 )
      goto LABEL_8;
    if ( !(unsigned __int8)sub_E0D140(a1, &v12) )
    {
      *a2 = 0;
      a2[1] = 0;
    }
    v8 = v12 == 0;
    *(_DWORD *)(a1 + 16) = v6;
    if ( v8 )
    {
LABEL_8:
      *a2 = 0;
      a2[1] = 0;
    }
    return 1;
  }
  else
  {
    if ( *v3 != 105 )
    {
      *a2 = 0;
LABEL_5:
      a2[1] = 0;
      return 0;
    }
    a2[1] = (__int64)(v3 + 1);
    *a2 = v2 - 1;
    return 1;
  }
}
