// Function: sub_883800
// Address: 0x883800
//
__int64 __fastcall sub_883800(__int64 a1, __int64 a2)
{
  __int64 v2; // rbp
  unsigned __int8 *v3; // rdi
  __int64 v4; // rax
  _QWORD v6[3]; // [rsp-18h] [rbp-18h] BYREF

  v3 = *(unsigned __int8 **)(a1 + 136);
  if ( !v3 )
    return 0;
  v6[2] = v2;
  v6[0] = a2;
  v6[1] = 0;
  v4 = sub_881B20(v3, (__int64)v6, 0);
  if ( v4 )
    return *(_QWORD *)(*(_QWORD *)v4 + 8LL);
  else
    return 0;
}
