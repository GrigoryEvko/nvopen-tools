// Function: sub_8C6CA0
// Address: 0x8c6ca0
//
__int64 __fastcall sub_8C6CA0(__int64 a1, __int64 a2, unsigned __int8 a3, _QWORD *a4)
{
  unsigned int v4; // ebx
  __int64 result; // rax
  __int64 v6; // r15
  __int64 *v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 i; // rax
  _QWORD *v11; // rax
  __int64 v12; // r8
  __int64 v13; // [rsp+0h] [rbp-400h]
  __int64 v14; // [rsp+8h] [rbp-3F8h]
  __int64 v15; // [rsp+8h] [rbp-3F8h]
  _QWORD *v16; // [rsp+8h] [rbp-3F8h]
  int v19; // [rsp+24h] [rbp-3DCh] BYREF
  __int64 (__fastcall *v20)(__int64, __int64, _QWORD, __int64, __int64); // [rsp+28h] [rbp-3D8h] BYREF
  _QWORD v21[122]; // [rsp+30h] [rbp-3D0h] BYREF

  v4 = a3;
  result = sub_5CEB70(a1, a3);
  v6 = *(_QWORD *)result;
  if ( *(_QWORD *)result )
  {
    v7 = *(__int64 **)sub_5CEB70(a2, v4);
    result = 0;
    memset(v21, 0, 0x398u);
    for ( ; v7; v7 = (__int64 *)*v7 )
    {
      result = *((unsigned __int8 *)v7 + 8);
      if ( !v21[result] || (*((_BYTE *)v7 + 11) & 1) != 0 )
        v21[result] = v7;
    }
    do
    {
      while ( 1 )
      {
        if ( *(_BYTE *)(v6 + 8) <= 1u || (*(_BYTE *)(v6 + 11) & 0x20) != 0 )
          goto LABEL_14;
        sub_5CB700(v6, v4, &v19, &v20);
        result = v19 & 3;
        v9 = v21[*(unsigned __int8 *)(v6 + 8)];
        if ( (_DWORD)result == 3 )
        {
          result = v20(a1, a2, v4, v6, v9);
          goto LABEL_17;
        }
        if ( v9 )
          break;
        if ( (v19 & 3) == 0 )
        {
          result = *(unsigned int *)a4;
          if ( (_DWORD)result )
          {
            v16 = sub_67D9E0(0x76Au, (_DWORD *)(v6 + 56), *(_QWORD *)(v6 + 16));
            sub_67DDB0(v16, 1062, a4);
            result = sub_685910((__int64)v16, (FILE *)0x426);
            goto LABEL_17;
          }
        }
LABEL_13:
        *(_BYTE *)(v6 + 11) |= 4u;
        if ( (*(_BYTE *)(v6 + 11) & 4) != 0 )
          goto LABEL_18;
LABEL_14:
        v6 = *(_QWORD *)v6;
        if ( !v6 )
          return result;
      }
      v14 = v21[*(unsigned __int8 *)(v6 + 8)];
      result = sub_5CB890(v6, v9, 0, v8);
      if ( !(_DWORD)result )
      {
        result = v19 & 3;
        if ( (_DWORD)result != 2 )
        {
          v11 = sub_67D9E0(0x76Bu, (_DWORD *)(v6 + 56), *(_QWORD *)(v6 + 16));
          v12 = v14;
          v15 = (__int64)v11;
          v13 = v12;
          sub_67DDB0(v11, 1062, (_QWORD *)(v12 + 56));
          result = sub_685910(v15, (FILE *)0x426);
          *(_BYTE *)(v6 + 8) = 0;
          *(_BYTE *)(v13 + 8) = 0;
          goto LABEL_17;
        }
        goto LABEL_13;
      }
      if ( (v19 & 4) != 0 )
        goto LABEL_13;
LABEL_17:
      if ( (*(_BYTE *)(v6 + 11) & 4) == 0 )
        goto LABEL_14;
LABEL_18:
      if ( a3 != 8 )
        goto LABEL_14;
      for ( i = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      result = *(_QWORD *)(*(_QWORD *)i + 96LL);
      *(_BYTE *)(result + 181) |= 0x40u;
      v6 = *(_QWORD *)v6;
    }
    while ( v6 );
  }
  return result;
}
