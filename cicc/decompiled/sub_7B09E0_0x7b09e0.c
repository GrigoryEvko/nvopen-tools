// Function: sub_7B09E0
// Address: 0x7b09e0
//
__int64 __fastcall sub_7B09E0(
        char *a1,
        int a2,
        __int64 a3,
        int a4,
        int a5,
        int a6,
        int a7,
        int a8,
        char **a9,
        char **a10,
        __int64 *a11,
        _DWORD *a12,
        _DWORD *a13,
        _QWORD *a14)
{
  char *v14; // r14
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 result; // rax
  char *v21; // [rsp+10h] [rbp-40h] BYREF
  int v22[14]; // [rsp+18h] [rbp-38h] BYREF

  v14 = a1;
  *a14 = 0;
  *a13 = 0;
  sub_720D70(v22);
  if ( a2 )
  {
    if ( a5 && (v17 = *(_QWORD *)(unk_4F064B0 + 32LL)) != 0 )
    {
      v18 = *(_QWORD *)(v17 + 16);
    }
    else if ( a4 )
    {
      v18 = unk_4F07688;
    }
    else
    {
      v18 = qword_4F076A8;
    }
  }
  else
  {
    v18 = 0;
  }
  *a11 = 0;
  *a12 = 0;
  *a9 = 0;
  if ( !unk_4F064B0 && *a1 == 45 && !a1[1] )
  {
    *a11 = (__int64)stdin;
    result = 1;
    goto LABEL_10;
  }
  if ( a6 )
  {
    result = sub_7AFFB0(a1, a2, v18, (__int64 **)qword_4F084F8, a6, a7, &v21, a11, a12, v22, a13, a14, 0);
    v14 = v21;
    if ( (_DWORD)result )
      goto LABEL_10;
  }
  else
  {
    result = sub_7AFFB0(a1, a2, v18, (__int64 **)qword_4F084F0, 0, a7, &v21, a11, a12, v22, a13, a14, 0);
    if ( (_DWORD)result )
    {
      v14 = v21;
LABEL_10:
      *a10 = v14;
      *a9 = v14;
      return result;
    }
    if ( a8 )
      sub_685AD0(7u, 1702, (__int64)a1, v22);
    else
      sub_685AD0(9 - ((a7 == 0) - 1), 1702, (__int64)a1, v22);
    return 0;
  }
  return result;
}
