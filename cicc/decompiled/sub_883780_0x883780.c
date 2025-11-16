// Function: sub_883780
// Address: 0x883780
//
__int64 __fastcall sub_883780(int a1)
{
  __int64 v1; // rbp
  __int64 v2; // rax
  unsigned __int8 *v3; // r8
  __int64 v4; // rax
  int v6; // [rsp-18h] [rbp-18h] BYREF
  __int64 v7; // [rsp-8h] [rbp-8h]

  if ( dword_4F04C38 )
    return 0;
  v2 = *(_QWORD *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 408);
  if ( !v2 )
    return 0;
  v3 = *(unsigned __int8 **)(v2 + 88);
  if ( !v3 )
    return 0;
  v7 = v1;
  v6 = a1;
  v4 = sub_881B20(v3, (__int64)&v6, 0);
  if ( v4 )
    return *(_QWORD *)(*(_QWORD *)v4 + 8LL);
  else
    return 0;
}
