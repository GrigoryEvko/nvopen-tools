// Function: sub_2DF9F10
// Address: 0x2df9f10
//
__int64 __fastcall sub_2DF9F10(__int64 a1, __int64 a2, _QWORD *a3)
{
  int v4; // r15d
  __int64 result; // rax
  bool v6; // zf
  __int64 v7; // r14
  int v8; // r15d
  int v9; // eax
  __int64 v10; // rcx
  __int64 v11; // r8
  int v12; // r9d
  unsigned int i; // edx
  __int64 v14; // r12
  bool v15; // al
  char v16; // di
  __int64 v17; // [rsp+0h] [rbp-A0h]
  int v18; // [rsp+8h] [rbp-98h]
  unsigned int v19; // [rsp+Ch] [rbp-94h]
  __int64 v20; // [rsp+10h] [rbp-90h]
  int v21; // [rsp+2Ch] [rbp-74h] BYREF
  __int64 v22; // [rsp+30h] [rbp-70h] BYREF
  __int64 v23; // [rsp+38h] [rbp-68h] BYREF
  _QWORD v24[12]; // [rsp+40h] [rbp-60h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *(_BYTE *)(a2 + 24) == 0;
  memset(v24, 0, 24);
  v7 = *(_QWORD *)(a1 + 8);
  v24[3] = 1;
  v24[4] = 0;
  v21 = 0;
  if ( !v6 )
    v21 = *(unsigned __int16 *)(a2 + 16) | (*(_DWORD *)(a2 + 8) << 16);
  v8 = v4 - 1;
  v23 = *(_QWORD *)(a2 + 32);
  v22 = *(_QWORD *)a2;
  v9 = sub_F11290(&v22, &v21, &v23);
  v10 = *(_QWORD *)a2;
  v11 = 0;
  v12 = 1;
  for ( i = v8 & v9; ; i = v8 & (v18 + v19) )
  {
    v14 = v7 + 48LL * i;
    if ( v10 != *(_QWORD *)v14 )
      break;
    v16 = *(_BYTE *)(a2 + 24);
    if ( v16 != *(_BYTE *)(v14 + 24) )
      break;
    if ( !v16 || *(_QWORD *)(a2 + 8) == *(_QWORD *)(v14 + 8) && *(_QWORD *)(a2 + 16) == *(_QWORD *)(v14 + 16) )
    {
      if ( *(_QWORD *)(a2 + 32) == *(_QWORD *)(v14 + 32) )
      {
        *a3 = v14;
        return 1;
      }
      break;
    }
LABEL_8:
    v17 = v10;
    v18 = v12;
    v19 = i;
    v20 = v11;
    v15 = sub_F34140(v7 + 48LL * i, (__int64)v24);
    v10 = v17;
    if ( v20 || !v15 )
      v14 = v20;
    v11 = v14;
    v12 = v18 + 1;
  }
  if ( *(_QWORD *)v14 )
    goto LABEL_8;
  result = *(unsigned __int8 *)(v14 + 24);
  if ( (_BYTE)result || *(_QWORD *)(v14 + 32) )
    goto LABEL_8;
  if ( !v11 )
    v11 = v7 + 48LL * i;
  *a3 = v11;
  return result;
}
