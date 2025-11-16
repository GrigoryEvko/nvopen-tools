// Function: sub_831A40
// Address: 0x831a40
//
__int64 __fastcall sub_831A40(__int64 a1, int a2, __int64 *a3, _QWORD *a4)
{
  __int64 v7; // r15
  __int64 v8; // r10
  __int64 v9; // rax
  __int64 v10; // rbx
  bool v11; // zf
  __int64 v12; // [rsp+0h] [rbp-50h]
  __int64 v13; // [rsp+8h] [rbp-48h]
  __int64 v14; // [rsp+10h] [rbp-40h] BYREF
  __int64 v15[7]; // [rsp+18h] [rbp-38h] BYREF

  *a3 = 0;
  *a4 = 0;
  if ( !(unsigned int)sub_8319F0(a1, &v14) )
  {
    if ( *(_BYTE *)(a1 + 16) == 1 )
    {
      v7 = *(_QWORD *)(a1 + 144);
      if ( *(_BYTE *)(v7 + 24) == 1 && *(_BYTE *)(v7 + 56) == 91 )
      {
        v8 = *(_QWORD *)(*(_QWORD *)(v7 + 72) + 16LL);
        if ( (unsigned __int8)(*(_BYTE *)(v8 + 24) - 5) <= 1u )
        {
          if ( *(_BYTE *)(*(_QWORD *)(v8 + 56) + 48LL) )
          {
            v12 = *(_QWORD *)(*(_QWORD *)(v7 + 72) + 16LL);
            v13 = *(_QWORD *)(v8 + 56);
            sub_733B20((_QWORD *)v13);
            *(_BYTE *)(v13 + 49) &= 0xEEu;
            if ( *(_QWORD *)(v13 + 16) )
              *(_QWORD *)(v13 + 16) = 0;
            v9 = sub_6ECAE0(*(_QWORD *)a1, 0, 0, 1, 3u, (__int64 *)(a1 + 68), v15);
            v10 = v15[0];
            v14 = v9;
            *(_BYTE *)(v15[0] + 51) |= 6u;
            *(_QWORD *)(v10 + 56) = v7;
            *(_QWORD *)(*(_QWORD *)(v12 + 56) + 112LL) = v10;
            if ( !*(_BYTE *)(v10 + 48) )
              sub_721090();
            goto LABEL_13;
          }
        }
      }
    }
    return 0;
  }
  v10 = *(_QWORD *)(v14 + 56);
  v11 = *(_BYTE *)(v10 + 48) == 0;
  v15[0] = v10;
  if ( v11 )
    return 0;
LABEL_13:
  sub_733B20((_QWORD *)v10);
  *(_BYTE *)(v10 + 49) &= 0xEEu;
  if ( a2 )
  {
    if ( *(_QWORD *)(v10 + 16) )
      *(_QWORD *)(v10 + 16) = 0;
  }
  *a3 = v14;
  *a4 = v15[0];
  return 1;
}
