// Function: sub_1DF7D80
// Address: 0x1df7d80
//
__int64 __fastcall sub_1DF7D80(__int64 a1, __int64 a2)
{
  int v2; // r13d
  __int64 v3; // r15
  int v5; // r13d
  int v6; // eax
  int v7; // ecx
  unsigned int i; // r14d
  __int64 *v9; // r12
  __int64 v10; // r8
  char v11; // al
  int v12; // [rsp+Ch] [rbp-44h]
  int v13; // [rsp+Ch] [rbp-44h]
  _QWORD v14[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = *(_DWORD *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  v14[0] = a2;
  if ( !v2 )
    return 0;
  v5 = v2 - 1;
  v6 = sub_1E1C690(v14);
  v7 = 1;
  for ( i = v5 & v6; ; i = v5 & (v13 + i) )
  {
    v9 = (__int64 *)(v3 + 16LL * i);
    v10 = *v9;
    if ( (unsigned __int64)(*v9 - 1) > 0xFFFFFFFFFFFFFFFDLL || (unsigned __int64)(v14[0] - 1LL) > 0xFFFFFFFFFFFFFFFDLL )
      break;
    v12 = v7;
    v11 = sub_1E15D60(v14[0], *v9, 3);
    v7 = v12;
    if ( v11 )
      goto LABEL_11;
    v10 = *v9;
LABEL_8:
    v13 = v7;
    if ( sub_1DF7390(v10, 0) )
      return 0;
    sub_1DF7390(*v9, -1);
    v7 = v13 + 1;
  }
  if ( v14[0] != v10 )
    goto LABEL_8;
LABEL_11:
  if ( v9 == (__int64 *)(*(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 24)) )
    return 0;
  return *(unsigned int *)(v9[1] + 24);
}
