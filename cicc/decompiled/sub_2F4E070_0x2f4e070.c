// Function: sub_2F4E070
// Address: 0x2f4e070
//
void __fastcall sub_2F4E070(__int64 a1, int a2, int a3)
{
  unsigned int v3; // edx
  __int64 v4; // rbx
  unsigned int v5; // edx
  unsigned __int64 v6; // rax
  int v7; // r14d
  int v8; // r15d
  __int64 v9; // r9
  _DWORD *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // [rsp-40h] [rbp-40h]

  v3 = a3 & 0x7FFFFFFF;
  if ( v3 < *(_DWORD *)(a1 + 928) )
  {
    v4 = v3;
    v5 = (a2 & 0x7FFFFFFF) + 1;
    *(_DWORD *)(*(_QWORD *)(a1 + 920) + 8 * v4) = 1;
    v6 = *(unsigned int *)(a1 + 928);
    if ( v5 > (unsigned int)v6 && v5 != v6 )
    {
      if ( v5 >= v6 )
      {
        v7 = *(_DWORD *)(a1 + 936);
        v8 = *(_DWORD *)(a1 + 940);
        v9 = v5 - v6;
        if ( v5 > (unsigned __int64)*(unsigned int *)(a1 + 932) )
        {
          v12 = v5 - v6;
          sub_C8D5F0(a1 + 920, (const void *)(a1 + 936), v5, 8u, v5, v9);
          v6 = *(unsigned int *)(a1 + 928);
          v9 = v12;
        }
        v10 = (_DWORD *)(*(_QWORD *)(a1 + 920) + 8 * v6);
        v11 = v9;
        do
        {
          if ( v10 )
          {
            *v10 = v7;
            v10[1] = v8;
          }
          v10 += 2;
          --v11;
        }
        while ( v11 );
        *(_DWORD *)(a1 + 928) += v9;
      }
      else
      {
        *(_DWORD *)(a1 + 928) = v5;
      }
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 920) + 8LL * (a2 & 0x7FFFFFFF)) = *(_QWORD *)(*(_QWORD *)(a1 + 920) + 8 * v4);
  }
}
