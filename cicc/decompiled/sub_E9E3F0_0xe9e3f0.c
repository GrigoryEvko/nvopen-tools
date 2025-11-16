// Function: sub_E9E3F0
// Address: 0xe9e3f0
//
__int64 __fastcall sub_E9E3F0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // r9
  unsigned int v9; // edi
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rcx
  __int64 v13; // rdx
  unsigned int v15; // [rsp+4h] [rbp-2Ch] BYREF
  unsigned int *v16; // [rsp+8h] [rbp-28h] BYREF

  v8 = a1[1];
  v9 = *(_DWORD *)(v8 + 1912);
  v10 = *(_QWORD *)(v8 + 1744);
  v11 = v8 + 1736;
  v15 = v9;
  if ( !v10 )
    goto LABEL_8;
  do
  {
    while ( 1 )
    {
      v12 = *(_QWORD *)(v10 + 16);
      v13 = *(_QWORD *)(v10 + 24);
      if ( v9 <= *(_DWORD *)(v10 + 32) )
        break;
      v10 = *(_QWORD *)(v10 + 24);
      if ( !v13 )
        goto LABEL_6;
    }
    v11 = v10;
    v10 = *(_QWORD *)(v10 + 16);
  }
  while ( v12 );
LABEL_6:
  if ( v8 + 1736 == v11 || v9 < *(_DWORD *)(v11 + 32) )
  {
LABEL_8:
    v16 = &v15;
    v11 = sub_E9E2A0((_QWORD *)(v8 + 1728), v11, &v16);
  }
  return sub_E7B400(v11 + 40, a1, a2, a3, a4);
}
