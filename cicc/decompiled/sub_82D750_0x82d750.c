// Function: sub_82D750
// Address: 0x82d750
//
_QWORD *__fastcall sub_82D750(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, _DWORD *a6)
{
  _QWORD *result; // rax
  _QWORD *v9; // rbx
  char v10; // al
  __int64 *v11; // rdi
  __int64 i; // rax
  int v13; // [rsp+4h] [rbp-4Ch] BYREF
  int v14; // [rsp+8h] [rbp-48h] BYREF
  int v15; // [rsp+Ch] [rbp-44h] BYREF
  _BYTE v16[64]; // [rsp+10h] [rbp-40h] BYREF

  result = (_QWORD *)sub_82C9F0(a1, a2, a3, 1, a4, 0, 0, &v13, (__int64)v16, 0, &v14, &v15);
  if ( result && (v9 = result, result = (_QWORD *)sub_8DD690(v16, 0, 0, 0, 0, 0), (_DWORD)result) && (v16[13] & 4) == 0 )
  {
    v10 = *((_BYTE *)v9 + 80);
    if ( v10 == 16 )
    {
      v9 = *(_QWORD **)v9[11];
      v10 = *((_BYTE *)v9 + 80);
    }
    if ( v10 == 24 )
      v9 = (_QWORD *)v9[11];
    v11 = (__int64 *)v9[11];
    for ( i = v11[19]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    if ( *(_QWORD *)(*(_QWORD *)(i + 168) + 40LL) )
      return sub_73F170(v11, a5);
    else
      return (_QWORD *)sub_72D3B0((__int64)v11, a5, 1);
  }
  else
  {
    *a6 = 1;
  }
  return result;
}
