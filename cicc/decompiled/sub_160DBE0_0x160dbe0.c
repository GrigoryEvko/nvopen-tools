// Function: sub_160DBE0
// Address: 0x160dbe0
//
__int64 __fastcall sub_160DBE0(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // r14
  __int64 (__fastcall *v4)(__int64, __int64); // rax
  int v5; // r13d
  int v6; // ebx
  __int64 v7; // r15
  __int64 v8; // rdi
  __int64 *v9; // rbx
  __int64 *i; // r13
  __int64 v11; // rdi
  __int64 (*v12)(); // rax
  __int64 v15; // [rsp+10h] [rbp-40h]
  unsigned __int8 v16; // [rsp+1Bh] [rbp-35h]
  int v17; // [rsp+1Ch] [rbp-34h]

  v2 = *(_DWORD *)(a1 + 608) - 1;
  v17 = v2;
  if ( v2 < 0 )
  {
    v16 = 0;
  }
  else
  {
    v16 = 0;
    v15 = 8LL * v2;
    do
    {
      v3 = *(_QWORD *)(*(_QWORD *)(a1 + 600) + v15);
      if ( !v3 )
        BUG();
      v4 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)(v3 - 160) + 32LL);
      if ( v4 == sub_160CB70 )
      {
        v5 = *(_DWORD *)(v3 + 32) - 1;
        if ( v5 >= 0 )
        {
          v6 = 0;
          v7 = 8LL * v5;
          do
          {
            --v5;
            v8 = *(_QWORD *)(*(_QWORD *)(v3 + 24) + v7);
            v7 -= 8;
            v6 |= (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v8 + 32LL))(v8, a2);
          }
          while ( v5 != -1 );
          v16 |= v6;
        }
      }
      else
      {
        v16 |= v4(v3 - 160, a2);
      }
      --v17;
      v15 -= 8;
    }
    while ( v17 != -1 );
  }
  v9 = *(__int64 **)(a1 + 824);
  for ( i = &v9[*(unsigned int *)(a1 + 832)]; i != v9; v16 |= ((__int64 (__fastcall *)(__int64, __int64))v12)(v11, a2) )
  {
    while ( 1 )
    {
      v11 = *v9;
      v12 = *(__int64 (**)())(*(_QWORD *)*v9 + 32LL);
      if ( v12 != sub_134C080 )
        break;
      if ( i == ++v9 )
        return v16;
    }
    ++v9;
  }
  return v16;
}
