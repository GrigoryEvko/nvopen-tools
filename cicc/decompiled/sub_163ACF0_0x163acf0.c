// Function: sub_163ACF0
// Address: 0x163acf0
//
__int64 __fastcall sub_163ACF0(__int64 a1, const char *a2, _QWORD *a3)
{
  __int64 v3; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r14
  const void *v9; // r15
  size_t v10; // rax
  __int64 v11; // rdx
  _QWORD *v12; // rax

  v3 = *(_QWORD *)(a1 - 8);
  if ( **(_BYTE **)(a1 - 16) )
    return 0;
  if ( *(_BYTE *)v3 != 1 )
    return 0;
  v6 = sub_161E970(*(_QWORD *)(a1 - 16));
  v8 = v7;
  v9 = (const void *)v6;
  v10 = strlen(a2);
  if ( v10 != v8 || v10 && memcmp(v9, a2, v10) )
    return 0;
  v11 = *(_QWORD *)(v3 + 136);
  v12 = *(_QWORD **)(v11 + 24);
  if ( *(_DWORD *)(v11 + 32) > 0x40u )
    v12 = (_QWORD *)*v12;
  *a3 = v12;
  return 1;
}
