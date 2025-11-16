// Function: sub_3175680
// Address: 0x3175680
//
__int64 __fastcall sub_3175680(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 result; // rax
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned __int64 v8; // rax
  __int64 v9; // rbx
  __int64 i; // r15
  unsigned __int64 v11; // rcx
  _BYTE *v12; // rsi
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // [rsp+8h] [rbp-98h]
  __int64 v18; // [rsp+18h] [rbp-88h]
  __int64 *v19; // [rsp+20h] [rbp-80h] BYREF
  __int64 v20; // [rsp+28h] [rbp-78h]
  _BYTE v21[112]; // [rsp+30h] [rbp-70h] BYREF

  v3 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)a2 == 85 )
  {
    if ( !v3 || *(_BYTE *)v3 )
      return 0;
    v5 = *(_QWORD *)(v3 + 24);
    if ( v5 == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v3 + 33) & 0x20) != 0 && *(_DWORD *)(v3 + 36) == 336 )
      return *(_QWORD *)(*(_QWORD *)(a1 + 240) + 8LL);
  }
  else
  {
    if ( !v3 || *(_BYTE *)v3 )
      return 0;
    v5 = *(_QWORD *)(v3 + 24);
  }
  if ( v5 != *(_QWORD *)(a2 + 80) || !sub_971E80(a2, *(_QWORD *)(a2 - 32)) )
    return 0;
  v19 = (__int64 *)v21;
  v20 = 0x800000000LL;
  v8 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( v8 > 8 )
  {
    sub_C8D5F0((__int64)&v19, v21, v8, 8u, v6, v7);
    v8 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  }
  if ( (_DWORD)v8 == 1 )
  {
    v11 = (unsigned int)v20;
LABEL_26:
    result = sub_97A150(a2, v3, v19, v11, 0, 1);
  }
  else
  {
    v9 = (unsigned int)(v8 - 2);
    for ( i = 0; ; ++i )
    {
      v12 = *(_BYTE **)(a2 + 32 * (i - v8));
      if ( *v12 == 24 )
        break;
      v13 = sub_31751A0(a1, v12);
      if ( !v13 )
        break;
      v16 = (unsigned int)v20;
      if ( (unsigned __int64)(unsigned int)v20 + 1 > HIDWORD(v20) )
      {
        v17 = v13;
        sub_C8D5F0((__int64)&v19, v21, (unsigned int)v20 + 1LL, 8u, v14, v15);
        v16 = (unsigned int)v20;
        v13 = v17;
      }
      v19[v16] = v13;
      v11 = (unsigned int)(v20 + 1);
      LODWORD(v20) = v20 + 1;
      if ( v9 == i )
        goto LABEL_26;
      v8 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    }
    result = 0;
  }
  if ( v19 != (__int64 *)v21 )
  {
    v18 = result;
    _libc_free((unsigned __int64)v19);
    return v18;
  }
  return result;
}
