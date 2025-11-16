// Function: sub_2E97850
// Address: 0x2e97850
//
__int64 __fastcall sub_2E97850(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE **v4; // rsi
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 *v9; // rbx
  __int64 *v10; // r13
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 result; // rax
  __int64 *v14; // [rsp+10h] [rbp-80h] BYREF
  __int64 v15; // [rsp+18h] [rbp-78h]
  _BYTE v16[112]; // [rsp+20h] [rbp-70h] BYREF

  if ( **(_QWORD **)(a3 + 32) == a2 )
    goto LABEL_11;
  v4 = (_BYTE **)&v14;
  v14 = (__int64 *)v16;
  v15 = 0x800000000LL;
  sub_2EA42C0(a3, &v14);
  v9 = v14;
  v10 = &v14[(unsigned int)v15];
  if ( v14 == v10 )
  {
LABEL_9:
    if ( v10 != (__int64 *)v16 )
      _libc_free((unsigned __int64)v10);
LABEL_11:
    *(_DWORD *)(a1 + 1456) = 0;
    return 1;
  }
  while ( 1 )
  {
    v11 = *v9;
    v12 = sub_2E6AFC0(*(_QWORD *)(a1 + 384), (__int64)v4, v5, v6, v7, v8);
    v4 = (_BYTE **)a2;
    result = sub_2E6D360(v12, a2, v11);
    if ( !(_BYTE)result )
      break;
    if ( v10 == ++v9 )
    {
      v10 = v14;
      goto LABEL_9;
    }
  }
  *(_DWORD *)(a1 + 1456) = 1;
  if ( v14 != (__int64 *)v16 )
  {
    _libc_free((unsigned __int64)v14);
    return 0;
  }
  return result;
}
