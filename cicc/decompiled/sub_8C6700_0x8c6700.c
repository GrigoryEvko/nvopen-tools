// Function: sub_8C6700
// Address: 0x8c6700
//
__int64 __fastcall sub_8C6700(__int64 *a1, unsigned int *a2, unsigned int a3, unsigned int a4)
{
  __int64 v6; // r14
  __int64 v7; // rbx
  unsigned int *v8; // r15
  __int64 v9; // rax
  const char *v10; // rdi
  const char *v11; // rsi
  __int64 result; // rax
  _QWORD *v13; // r14
  __int64 v14; // rax
  const char *v15; // rax
  char *v16; // rax
  __int64 v18; // [rsp+8h] [rbp-48h]
  int v19; // [rsp+14h] [rbp-3Ch] BYREF
  int v20; // [rsp+18h] [rbp-38h] BYREF
  _DWORD v21[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v6 = *a1;
  v7 = qword_4F074B0;
  v18 = sub_729B10(*(_DWORD *)(*a1 + 48), &v19, v21, 1);
  v8 = (unsigned int *)(v6 + 48);
  v9 = sub_729B10(*a2, &v20, v21, 1);
  if ( v18 && v9 && (v10 = *(const char **)(v18 + 8)) != 0 && (v11 = *(const char **)(v9 + 8)) != 0 && !strcmp(v10, v11) )
  {
    if ( !(unsigned int)sub_67D520(a3, 8u, (unsigned int *)(v6 + 48)) )
    {
      v14 = sub_729AB0(*a2);
      v15 = (const char *)sub_723640(v14, 1, 0);
      v16 = sub_724840(0, v15);
      sub_686A10(a3, v8, (__int64)v16, v6);
      sub_67D470(a3, 8u, v8);
    }
  }
  else if ( !(unsigned int)sub_67D520(a4, 8u, (unsigned int *)(v6 + 48)) )
  {
    v13 = sub_67E020(a4, (_DWORD *)(v6 + 48), v6);
    sub_67DDB0(v13, 1062, a2);
    sub_685910((__int64)v13, (FILE *)0x426);
    sub_67D470(a4, 8u, v8);
  }
  result = qword_4F074B0 + qword_4F60258 - v7;
  qword_4F60258 = result;
  return result;
}
