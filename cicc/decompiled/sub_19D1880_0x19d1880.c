// Function: sub_19D1880
// Address: 0x19d1880
//
__int64 __fastcall sub_19D1880(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  int v6; // edx
  __int64 v7; // rdi
  unsigned __int8 *v8; // rdx
  char v9; // r8
  __int64 result; // rax
  __int64 v11; // rcx
  __int64 v12; // r15
  _QWORD *v13; // rdx
  unsigned __int64 v14; // rcx
  __int64 v15; // rax
  unsigned __int8 *v16; // rsi
  __int64 v17; // rax
  unsigned int v18; // eax
  unsigned __int8 *v19[6]; // [rsp+0h] [rbp-B0h] BYREF
  unsigned __int8 *v20; // [rsp+30h] [rbp-80h] BYREF
  __int64 v21; // [rsp+38h] [rbp-78h]
  __int64 v22; // [rsp+40h] [rbp-70h]
  __int64 v23; // [rsp+48h] [rbp-68h]
  __int64 v24; // [rsp+50h] [rbp-60h]
  int v25; // [rsp+58h] [rbp-58h]
  __int64 v26; // [rsp+60h] [rbp-50h]
  __int64 v27; // [rsp+68h] [rbp-48h]

  if ( !*(_QWORD *)(a1 + 32) )
    sub_4263D6(a1, a2, a3);
  v5 = (*(__int64 (__fastcall **)(__int64))(a1 + 40))(a1 + 16);
  v6 = *(_DWORD *)(a2 + 20);
  v7 = v5;
  v21 = 1;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v19[1] = (unsigned __int8 *)1;
  v8 = *(unsigned __int8 **)(a2 + 24 * (1LL - (v6 & 0xFFFFFFF)));
  memset(&v19[2], 0, 24);
  LODWORD(v5) = *(_DWORD *)(a3 + 20);
  v20 = v8;
  v19[0] = *(unsigned __int8 **)(a3 - 24 * (v5 & 0xFFFFFFF));
  v9 = sub_134CB50(v7, (__int64)v19, (__int64)&v20);
  result = 0;
  if ( v9 == 3 )
  {
    v11 = *(_QWORD *)(a3 + 24 * (2LL - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)));
    if ( *(_BYTE *)(v11 + 16) == 13 )
    {
      v12 = *(_QWORD *)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      v13 = *(_QWORD **)(v12 + 24);
      if ( *(_DWORD *)(v12 + 32) > 0x40u )
        v13 = (_QWORD *)*v13;
      if ( *(_DWORD *)(v11 + 32) <= 0x40u )
        v14 = *(_QWORD *)(v11 + 24);
      else
        v14 = **(_QWORD **)(v11 + 24);
      result = 0;
      if ( v14 >= (unsigned __int64)v13 )
      {
        v15 = sub_16498A0(a2);
        v16 = *(unsigned __int8 **)(a2 + 48);
        v20 = 0;
        v23 = v15;
        v17 = *(_QWORD *)(a2 + 40);
        v24 = 0;
        v21 = v17;
        v25 = 0;
        v26 = 0;
        v27 = 0;
        v22 = a2 + 24;
        v19[0] = v16;
        if ( v16 )
        {
          sub_1623A60((__int64)v19, (__int64)v16, 2);
          if ( v20 )
            sub_161E7C0((__int64)&v20, (__int64)v20);
          v20 = v19[0];
          if ( v19[0] )
            sub_1623210((__int64)v19, v19[0], (__int64)&v20);
        }
        v18 = sub_15603A0((_QWORD *)(a2 + 56), 0);
        sub_15E7280(
          (__int64 *)&v20,
          *(_QWORD **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)),
          *(_QWORD *)(a3 + 24 * (1LL - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF))),
          (__int64 *)v12,
          v18,
          0,
          0,
          0,
          0);
        if ( v20 )
        {
          sub_161E7C0((__int64)&v20, (__int64)v20);
          return 1;
        }
        else
        {
          return 1;
        }
      }
    }
  }
  return result;
}
