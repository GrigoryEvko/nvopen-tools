// Function: sub_881D30
// Address: 0x881d30
//
void __fastcall sub_881D30(__int64 *a1, __int64 a2)
{
  unsigned __int8 *v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 *v7; // rax
  _QWORD *v8; // rdx
  __int64 *v9; // rdx
  _QWORD v10[3]; // [rsp+0h] [rbp-20h] BYREF

  v3 = *(unsigned __int8 **)(a2 + 136);
  if ( v3 )
  {
    v4 = *a1;
    v10[1] = 0;
    v10[0] = v4;
    v5 = sub_881B20(v3, (__int64)v10, 0);
    v6 = *(_QWORD *)v5;
    v7 = *(__int64 **)(*(_QWORD *)v5 + 8LL);
    if ( a1 != v7 && v7 )
    {
      do
      {
        v9 = v7;
        v7 = (__int64 *)v7[4];
      }
      while ( a1 != v7 && v7 );
      v8 = v9 + 4;
    }
    else
    {
      v8 = (_QWORD *)(v6 + 8);
    }
    *v8 = a1[4];
    a1[4] = 0;
  }
}
