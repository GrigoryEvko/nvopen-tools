// Function: sub_18DC2F0
// Address: 0x18dc2f0
//
__int64 __fastcall sub_18DC2F0(__int64 a1, __int64 a2, __int64 *a3, unsigned int a4)
{
  unsigned int v7; // r8d
  unsigned __int8 v8; // al
  __int64 v9; // rsi
  __int64 v10; // rdi
  char v11; // al
  __int64 v13; // rax
  __int64 *v14; // r15
  _BYTE *v15; // r13
  unsigned __int8 v16; // al
  __int64 v17; // r14
  _BYTE *v18; // rax
  __int64 v19; // [rsp+8h] [rbp-88h]
  unsigned __int64 v20; // [rsp+18h] [rbp-78h]
  __int64 v21; // [rsp+28h] [rbp-68h] BYREF
  _BYTE *v22; // [rsp+30h] [rbp-60h] BYREF
  __int64 v23; // [rsp+38h] [rbp-58h]
  __int64 v24; // [rsp+40h] [rbp-50h]
  __int64 v25; // [rsp+48h] [rbp-48h]
  __int64 v26; // [rsp+50h] [rbp-40h]

  if ( a4 > 0x17 || (v7 = 0, ((1LL << a4) & 0x900060) == 0) )
  {
    v8 = *(_BYTE *)(a1 + 16);
    v9 = 0;
    if ( v8 > 0x17u )
    {
      if ( v8 == 78 )
      {
        v9 = a1 | 4;
      }
      else if ( v8 == 29 )
      {
        v9 = a1 & 0xFFFFFFFFFFFFFFFBLL;
      }
    }
    v10 = *a3;
    v21 = v9;
    v11 = sub_134CC90(v10, v9);
    if ( (v11 & 2) != 0 )
    {
      if ( (v11 & 0x30) != 0 )
        return 1;
      v13 = sub_15F2050(a1);
      v19 = sub_1632FA0(v13);
      v14 = (__int64 *)((v21 & 0xFFFFFFFFFFFFFFF8LL)
                      - 24LL * (*(_DWORD *)((v21 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
      v20 = sub_134EF80(&v21);
      if ( (__int64 *)v20 != v14 )
      {
        while ( 1 )
        {
          while ( 1 )
          {
            v15 = (_BYTE *)*v14;
            v16 = *(_BYTE *)(*v14 + 16);
            if ( v16 > 0x10u && v16 != 53 )
            {
              v17 = *a3;
              if ( (v16 != 17
                 || !(unsigned __int8)sub_15E0450(*v14)
                 && !(unsigned __int8)sub_15E0470((__int64)v15)
                 && !(unsigned __int8)sub_15E0490((__int64)v15)
                 && !(unsigned __int8)sub_15E04F0((__int64)v15))
                && *(_BYTE *)(*(_QWORD *)v15 + 8LL) == 15 )
              {
                v22 = v15;
                v23 = -1;
                v24 = 0;
                v25 = 0;
                v26 = 0;
                if ( !(unsigned __int8)sub_134CBB0(v17, (__int64)&v22, 0) )
                {
                  if ( v15[16] != 54 )
                    break;
                  v18 = (_BYTE *)*((_QWORD *)v15 - 3);
                  v23 = -1;
                  v22 = v18;
                  v24 = 0;
                  v25 = 0;
                  v26 = 0;
                  if ( !(unsigned __int8)sub_134CBB0(v17, (__int64)&v22, 0) )
                    break;
                }
              }
            }
            v14 += 3;
            if ( (__int64 *)v20 == v14 )
              return 0;
          }
          if ( (unsigned __int8)sub_18DDD00(a3, a2, v15, v19) )
            break;
          v14 += 3;
          if ( (__int64 *)v20 == v14 )
            return 0;
        }
        return 1;
      }
    }
    return 0;
  }
  return v7;
}
