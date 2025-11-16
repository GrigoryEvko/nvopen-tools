// Function: sub_3135D90
// Address: 0x3135d90
//
__int64 __fastcall sub_3135D90(__int64 a1, __int64 a2)
{
  _BYTE *v2; // rax
  unsigned __int64 v3; // rsi
  __int64 v5; // [rsp+8h] [rbp-48h] BYREF
  const char *v6; // [rsp+10h] [rbp-40h] BYREF
  char v7; // [rsp+30h] [rbp-20h]
  char v8; // [rsp+31h] [rbp-1Fh]

  v5 = a2;
  v8 = 1;
  v6 = "omp_global_thread_num";
  v7 = 3;
  v2 = sub_3135910(a1, 5);
  v3 = 0;
  if ( v2 )
    v3 = *((_QWORD *)v2 + 3);
  return sub_921880((unsigned int **)(a1 + 512), v3, (int)v2, (int)&v5, 1, (__int64)&v6, 0);
}
