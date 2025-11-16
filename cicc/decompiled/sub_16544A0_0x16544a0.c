// Function: sub_16544A0
// Address: 0x16544a0
//
__int64 *__fastcall sub_16544A0(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  unsigned int v6; // ebx
  int v8; // r13d
  unsigned int v9; // r14d
  __int64 *result; // rax
  __int64 v11; // rbx
  __int64 *v12; // rdi
  __int64 v13; // r12
  _BYTE *v14; // rax
  unsigned int v15; // [rsp+4h] [rbp-6Ch]
  __int64 v17; // [rsp+18h] [rbp-58h]
  __int64 *v18; // [rsp+18h] [rbp-58h]
  const char *v19; // [rsp+20h] [rbp-50h] BYREF
  char v20; // [rsp+30h] [rbp-40h]
  char v21; // [rsp+31h] [rbp-3Fh]

  v6 = *(_DWORD *)(a3 + 8);
  if ( v6 == 2 )
    return *(__int64 **)(a3 - 8);
  v17 = v6;
  v15 = a5 == 0 ? 1 : 3;
  v8 = 3 - (a5 == 0);
  if ( v6 <= v15 )
  {
LABEL_10:
    v11 = v6 - v8;
    sub_16A7590(a4, *(_QWORD *)(*(_QWORD *)(a3 + 8 * ((unsigned int)(v11 + 1) - v17)) + 136LL) + 24LL);
    return *(__int64 **)(a3 + 8 * (v11 - *(unsigned int *)(a3 + 8)));
  }
  v9 = a5 == 0 ? 1 : 3;
  while ( (int)sub_16A9900(*(_QWORD *)(*(_QWORD *)(a3 + 8 * (v9 + 1 - (unsigned __int64)v6)) + 136LL) + 24LL, a4) <= 0 )
  {
    v9 += v8;
    if ( v6 <= v9 )
      goto LABEL_10;
  }
  if ( v15 == v9 )
  {
    result = *a1;
    if ( *a1 )
    {
      v12 = *a1;
      v18 = *a1;
      v21 = 1;
      v19 = "Could not find TBAA parent in struct type node";
      v20 = 3;
      sub_164FF40(v12, (__int64)&v19);
      if ( *v18 )
      {
        sub_164FA80(v18, a2);
        sub_164ED40(v18, (unsigned __int8 *)a3);
        v13 = *v18;
        sub_16A95F0(a4, *v18, 1);
        v14 = *(_BYTE **)(v13 + 24);
        if ( (unsigned __int64)v14 >= *(_QWORD *)(v13 + 16) )
        {
          sub_16E7DE0(v13, 10);
          return 0;
        }
        *(_QWORD *)(v13 + 24) = v14 + 1;
        *v14 = 10;
      }
      return 0;
    }
  }
  else
  {
    sub_16A7590(a4, *(_QWORD *)(*(_QWORD *)(a3 + 8 * (v9 - v8 + 1 - (unsigned __int64)v6)) + 136LL) + 24LL);
    return *(__int64 **)(a3 + 8 * (v9 - v8 - (unsigned __int64)*(unsigned int *)(a3 + 8)));
  }
  return result;
}
