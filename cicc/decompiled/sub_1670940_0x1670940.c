// Function: sub_1670940
// Address: 0x1670940
//
__int64 __fastcall sub_1670940(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  int v4; // ebx
  __int64 v5; // r15
  int v7; // ebx
  __int64 *v8; // r14
  __int64 v9; // [rsp+8h] [rbp-68h]
  __int64 v10; // [rsp+10h] [rbp-60h]
  int v11; // [rsp+18h] [rbp-58h]
  unsigned int v12; // [rsp+1Ch] [rbp-54h]
  _BYTE v13[80]; // [rsp+20h] [rbp-50h] BYREF

  sub_1670470((__int64)v13, a2, a3, a4);
  v4 = *(_DWORD *)(a1 + 56);
  v5 = *(_QWORD *)(a1 + 40);
  if ( !v4 )
    return 0;
  v7 = v4 - 1;
  v10 = sub_16704E0();
  v9 = sub_16704F0();
  v11 = 1;
  v12 = v7 & sub_1670770((__int64)v13);
  while ( 1 )
  {
    v8 = (__int64 *)(v5 + 8LL * v12);
    if ( sub_1670500((__int64)v13, *v8) )
      break;
    if ( sub_1670560(*v8, v10) )
      return 0;
    sub_1670560(*v8, v9);
    v12 = v7 & (v11 + v12);
    ++v11;
  }
  if ( v8 == (__int64 *)(*(_QWORD *)(a1 + 40) + 8LL * *(unsigned int *)(a1 + 56)) )
    return 0;
  else
    return *v8;
}
