// Function: sub_2E4FF30
// Address: 0x2e4ff30
//
__int64 __fastcall sub_2E4FF30(__int64 a1, __int64 a2)
{
  int v2; // r15d
  __int64 v3; // r14
  int v4; // r15d
  int v5; // eax
  int v6; // ecx
  unsigned int i; // r13d
  __int64 *v8; // r12
  __int64 v9; // r8
  char v10; // al
  int v12; // [rsp+Ch] [rbp-44h]
  int v13; // [rsp+Ch] [rbp-44h]
  _QWORD v14[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = *(_DWORD *)(a1 + 128);
  v3 = *(_QWORD *)(a1 + 112);
  v14[0] = a2;
  if ( !v2 )
    return 0;
  v4 = v2 - 1;
  v5 = sub_2E8E920(v14);
  v6 = 1;
  for ( i = v4 & v5; ; i = v4 & (v13 + i) )
  {
    v8 = (__int64 *)(v3 + 16LL * i);
    v9 = *v8;
    if ( (unsigned __int64)(*v8 - 1) > 0xFFFFFFFFFFFFFFFDLL || (unsigned __int64)(v14[0] - 1LL) > 0xFFFFFFFFFFFFFFFDLL )
      break;
    v12 = v6;
    v10 = sub_2E88AF0(v14[0], *v8, 3);
    v6 = v12;
    if ( v10 )
      goto LABEL_6;
    v9 = *v8;
LABEL_9:
    v13 = v6;
    if ( sub_2E4F140(v9, 0) )
      return 0;
    v6 = v13 + 1;
  }
  if ( v14[0] != v9 )
    goto LABEL_9;
LABEL_6:
  if ( v8 != (__int64 *)(*(_QWORD *)(a1 + 112) + 16LL * *(unsigned int *)(a1 + 128)) )
    return *(unsigned int *)(v8[1] + 24);
  return 0;
}
