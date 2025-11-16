// Function: sub_1E1C9C0
// Address: 0x1e1c9c0
//
bool __fastcall sub_1E1C9C0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 *v4; // rbx
  __int64 v5; // r14
  __int64 v6; // r15
  bool result; // al
  __int64 *v8; // [rsp+8h] [rbp-88h]
  __int64 *v9; // [rsp+10h] [rbp-80h] BYREF
  __int64 v10; // [rsp+18h] [rbp-78h]
  _BYTE v11[112]; // [rsp+20h] [rbp-70h] BYREF

  v3 = *(_QWORD *)(a1 + 608);
  if ( **(_QWORD **)(v3 + 32) == a2 )
    goto LABEL_11;
  v9 = (__int64 *)v11;
  v10 = 0x800000000LL;
  sub_1E29B30(v3, &v9);
  v4 = v9;
  v8 = &v9[(unsigned int)v10];
  if ( v9 == v8 )
  {
LABEL_9:
    if ( v8 != (__int64 *)v11 )
      _libc_free((unsigned __int64)v8);
LABEL_11:
    *(_DWORD *)(a1 + 1848) = 0;
    return 1;
  }
  while ( 1 )
  {
    v5 = *(_QWORD *)(a1 + 592);
    v6 = *v4;
    sub_1E06620(v5);
    result = sub_1E05550(*(_QWORD *)(v5 + 1312), a2, v6);
    if ( !result )
      break;
    if ( v8 == ++v4 )
    {
      v8 = v9;
      goto LABEL_9;
    }
  }
  *(_DWORD *)(a1 + 1848) = 1;
  if ( v9 != (__int64 *)v11 )
  {
    _libc_free((unsigned __int64)v9);
    return 0;
  }
  return result;
}
