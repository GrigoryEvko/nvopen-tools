// Function: sub_130E080
// Address: 0x130e080
//
__int64 __fastcall sub_130E080(__int64 a1, __int64 a2, __int64 a3)
{
  int v5; // eax
  char v7; // r11
  __int64 v8; // rcx
  unsigned int v9; // edx
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rax
  char v14; // [rsp+7h] [rbp-9h] BYREF
  __int64 v15; // [rsp+8h] [rbp-8h] BYREF

  *(_QWORD *)(a3 + 128) = 0;
  v5 = *(_DWORD *)(a2 + 112);
  v15 = 0;
  if ( v5 )
  {
    v7 = 0;
    v8 = 0;
    v9 = 0;
    do
    {
      while ( 1 )
      {
        v11 = *(_QWORD *)(a3 + 120) + 24LL * v9;
        *(_QWORD *)(v11 + 8) = 0;
        if ( v8 )
          break;
        v8 = *(_QWORD *)(v11 + 16);
        v7 = 1;
        ++v9;
        *(_QWORD *)(v11 + 16) = 0;
        if ( *(_DWORD *)(a2 + 112) <= v9 )
          goto LABEL_8;
      }
      v10 = *(_QWORD *)(v11 + 16);
      if ( v10 )
      {
        *(_QWORD *)(*(_QWORD *)(v10 + 48) + 40LL) = *(_QWORD *)(v8 + 48);
        *(_QWORD *)(v8 + 48) = *(_QWORD *)(*(_QWORD *)(v11 + 16) + 48LL);
        *(_QWORD *)(*(_QWORD *)(v11 + 16) + 48LL) = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v11 + 16) + 48LL) + 40LL);
        *(_QWORD *)(*(_QWORD *)(v8 + 48) + 40LL) = v8;
        *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v11 + 16) + 48LL) + 40LL) = *(_QWORD *)(v11 + 16);
        *(_QWORD *)(v11 + 16) = 0;
      }
      ++v9;
    }
    while ( *(_DWORD *)(a2 + 112) > v9 );
LABEL_8:
    if ( v7 )
      v15 = v8;
  }
  v12 = *(_QWORD *)(a2 + 56);
  v14 = 0;
  return (*(__int64 (__fastcall **)(__int64, __int64, __int64 *, char *))(v12 + 40))(a1, v12, &v15, &v14);
}
