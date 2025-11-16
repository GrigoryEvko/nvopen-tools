// Function: sub_31758E0
// Address: 0x31758e0
//
__int64 __fastcall sub_31758E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // eax
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 i; // r14
  __int64 v10; // rdx
  __int64 result; // rax
  __int64 v12; // r8
  __int64 v13; // rdx
  unsigned __int64 v14; // r9
  __int64 v15; // [rsp+0h] [rbp-90h]
  __int64 v16; // [rsp+8h] [rbp-88h]
  __int64 *v17; // [rsp+10h] [rbp-80h] BYREF
  __int64 v18; // [rsp+18h] [rbp-78h]
  _BYTE v19[112]; // [rsp+20h] [rbp-70h] BYREF

  v18 = 0x800000000LL;
  v6 = *(_DWORD *)(a2 + 4);
  v17 = (__int64 *)v19;
  v7 = v6 & 0x7FFFFFF;
  if ( (unsigned int)v7 > 8uLL )
  {
    sub_C8D5F0((__int64)&v17, v19, (unsigned int)v7, 8u, a5, a6);
    v7 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  }
  if ( (_DWORD)v7 )
  {
    v8 = (unsigned int)(v7 - 1);
    for ( i = 0; ; ++i )
    {
      result = sub_31751A0(a1, *(_BYTE **)(a2 + 32 * (i - v7)));
      if ( !result )
        break;
      v13 = (unsigned int)v18;
      v14 = (unsigned int)v18 + 1LL;
      if ( v14 > HIDWORD(v18) )
      {
        v15 = result;
        sub_C8D5F0((__int64)&v17, v19, (unsigned int)v18 + 1LL, 8u, v12, v14);
        v13 = (unsigned int)v18;
        result = v15;
      }
      v17[v13] = result;
      v10 = (unsigned int)(v18 + 1);
      LODWORD(v18) = v18 + 1;
      if ( v8 == i )
        goto LABEL_11;
      v7 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    }
  }
  else
  {
    v10 = (unsigned int)v18;
LABEL_11:
    result = sub_97D230((unsigned __int8 *)a2, v17, v10, *(_BYTE **)(a1 + 40), 0, 1u);
  }
  if ( v17 != (__int64 *)v19 )
  {
    v16 = result;
    _libc_free((unsigned __int64)v17);
    return v16;
  }
  return result;
}
