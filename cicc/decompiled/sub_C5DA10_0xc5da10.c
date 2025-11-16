// Function: sub_C5DA10
// Address: 0xc5da10
//
_QWORD *__fastcall sub_C5DA10(__int64 a1)
{
  _QWORD *v2; // r13
  _QWORD *v3; // rax
  _QWORD *v4; // rsi
  __int64 v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // rax
  _QWORD *v8; // r9
  int v9; // edi
  __int64 v10; // r8
  __int64 v11; // rcx
  __int64 v12; // rdx
  _QWORD *result; // rax
  _QWORD *v14; // rdi
  unsigned __int64 v15; // [rsp+0h] [rbp-20h] BYREF
  int *v16[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = sub_C52410();
  v15 = sub_C959E0();
  v3 = (_QWORD *)v2[2];
  v4 = v2 + 1;
  if ( !v3 )
    goto LABEL_17;
  do
  {
    while ( 1 )
    {
      v5 = v3[2];
      v6 = v3[3];
      if ( v15 <= v3[4] )
        break;
      v3 = (_QWORD *)v3[3];
      if ( !v6 )
        goto LABEL_6;
    }
    v4 = v3;
    v3 = (_QWORD *)v3[2];
  }
  while ( v5 );
LABEL_6:
  if ( v2 + 1 == v4 || v15 < v4[4] )
  {
LABEL_17:
    v16[0] = (int *)&v15;
    v4 = (_QWORD *)sub_C5D700(v2, v4, (unsigned __int64 **)v16);
    v7 = v4[7];
    v8 = v4 + 6;
    if ( v7 )
      goto LABEL_9;
LABEL_18:
    v10 = (__int64)v8;
LABEL_19:
    v14 = v4 + 5;
    v4 = (_QWORD *)v10;
    v16[0] = (int *)(a1 + 8);
    v10 = sub_C5D7D0(v14, v10, v16);
    goto LABEL_15;
  }
  v7 = v4[7];
  v8 = v4 + 6;
  if ( !v7 )
    goto LABEL_18;
LABEL_9:
  v9 = *(_DWORD *)(a1 + 8);
  v10 = (__int64)v8;
  do
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)(v7 + 16);
      v12 = *(_QWORD *)(v7 + 24);
      if ( *(_DWORD *)(v7 + 32) >= v9 )
        break;
      v7 = *(_QWORD *)(v7 + 24);
      if ( !v12 )
        goto LABEL_13;
    }
    v10 = v7;
    v7 = *(_QWORD *)(v7 + 16);
  }
  while ( v11 );
LABEL_13:
  if ( v8 == (_QWORD *)v10 || v9 < *(_DWORD *)(v10 + 32) )
    goto LABEL_19;
LABEL_15:
  *(_DWORD *)(v10 + 36) = 0;
  result = (_QWORD *)(*(__int64 (__fastcall **)(__int64, _QWORD *, __int64, __int64))(*(_QWORD *)a1 + 64LL))(
                       a1,
                       v4,
                       v12,
                       v11);
  if ( (*(_BYTE *)(a1 + 13) & 0x20) != 0 )
    return sub_C53010(a1);
  return result;
}
