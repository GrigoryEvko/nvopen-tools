// Function: sub_1615900
// Address: 0x1615900
//
__int64 __fastcall sub_1615900(__int64 a1, const char *a2)
{
  __int64 *v3; // rbx
  __int64 *i; // r13
  __int64 v5; // rdi
  __int64 (*v6)(); // rax
  unsigned int j; // r13d
  __int64 v8; // r14
  __int64 (__fastcall *v9)(__int64, __int64); // rax
  int v10; // ebx
  unsigned int v11; // r15d
  __int64 v12; // rsi
  __int64 v13; // rdi
  unsigned __int8 v15; // [rsp+7h] [rbp-39h]

  sub_1615700(a1 + 568, a2);
  sub_160E900(a1 + 568);
  v3 = *(__int64 **)(a1 + 824);
  v15 = 0;
  for ( i = &v3[*(unsigned int *)(a1 + 832)];
        i != v3;
        v15 |= ((__int64 (__fastcall *)(__int64, const char *, __int64 (*)()))v6)(v5, a2, sub_134C070) )
  {
    while ( 1 )
    {
      v5 = *v3;
      v6 = *(__int64 (**)())(*(_QWORD *)*v3 + 24LL);
      if ( v6 != sub_134C070 )
        break;
      if ( i == ++v3 )
        goto LABEL_6;
    }
    ++v3;
  }
LABEL_6:
  for ( j = 0; j < *(_DWORD *)(a1 + 608); ++j )
  {
    v8 = *(_QWORD *)(*(_QWORD *)(a1 + 600) + 8LL * j);
    if ( !v8 )
      BUG();
    v9 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)(v8 - 160) + 24LL);
    if ( v9 == sub_160CB00 )
    {
      if ( *(_DWORD *)(v8 + 32) )
      {
        v10 = 0;
        v11 = 0;
        do
        {
          v12 = v11++;
          v13 = *(_QWORD *)(*(_QWORD *)(v8 + 24) + 8 * v12);
          v10 |= (*(__int64 (__fastcall **)(__int64, const char *))(*(_QWORD *)v13 + 24LL))(v13, a2);
        }
        while ( *(_DWORD *)(v8 + 32) > v11 );
        v15 |= v10;
      }
    }
    else
    {
      v15 |= v9(v8 - 160, (__int64)a2);
    }
  }
  return v15;
}
