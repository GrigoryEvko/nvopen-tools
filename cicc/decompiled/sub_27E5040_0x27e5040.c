// Function: sub_27E5040
// Address: 0x27e5040
//
__int64 __fastcall sub_27E5040(__int64 a1, __int64 a2)
{
  int v2; // r14d
  __int64 v3; // rcx
  unsigned int v4; // r14d
  __int64 v6; // r12
  __int64 v7; // r14
  __int64 v8; // rdi
  unsigned __int64 v9; // rax
  __int64 result; // rax
  _QWORD *v11; // rdi
  __int64 v12; // [rsp+8h] [rbp-58h]
  unsigned __int8 v13; // [rsp+8h] [rbp-58h]
  _QWORD v14[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v15[8]; // [rsp+20h] [rbp-40h] BYREF

  v2 = *(_DWORD *)(a2 + 4);
  v3 = *(_QWORD *)(a2 + 40);
  v14[0] = v15;
  v14[1] = 0x100000001LL;
  v15[0] = 0;
  v4 = v2 & 0x7FFFFFF;
  if ( !v4 )
    return 0;
  v6 = 0;
  v7 = 8LL * v4;
  do
  {
    v8 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL * *(unsigned int *)(a2 + 72) + v6);
    v9 = *(_QWORD *)(v8 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v9 == v8 + 48 )
      goto LABEL_16;
    if ( !v9 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v9 - 24) - 30 > 0xA )
LABEL_16:
      BUG();
    if ( *(_BYTE *)(v9 - 24) == 31 && (*(_DWORD *)(v9 - 20) & 0x7FFFFFF) == 1 )
    {
      v12 = v3;
      *(_QWORD *)v14[0] = v8;
      result = sub_27E4FC0(a1, v3, (__int64)v14);
      v3 = v12;
      if ( (_BYTE)result )
      {
        v11 = (_QWORD *)v14[0];
        goto LABEL_12;
      }
    }
    v6 += 8;
  }
  while ( v7 != v6 );
  v11 = (_QWORD *)v14[0];
  result = 0;
LABEL_12:
  if ( v11 != v15 )
  {
    v13 = result;
    _libc_free((unsigned __int64)v11);
    return v13;
  }
  return result;
}
