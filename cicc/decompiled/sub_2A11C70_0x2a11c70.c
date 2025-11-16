// Function: sub_2A11C70
// Address: 0x2a11c70
//
__int64 __fastcall sub_2A11C70(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        int a8,
        _QWORD *a9)
{
  __int64 v9; // rax
  __int64 result; // rax
  __int64 v11; // [rsp+8h] [rbp-48h]
  __int64 v12; // [rsp+10h] [rbp-40h]
  __int64 v13; // [rsp+18h] [rbp-38h]
  __int64 v14; // [rsp+20h] [rbp-30h]

  v9 = a9[7];
  if ( v9 != -4096 && v9 != 0 && v9 != -8192 )
    sub_BD60C0(a9 + 5);
  v11 = 2;
  v12 = 0;
  v13 = -8192;
  result = a9[3];
  v14 = 0;
  if ( result == -8192 )
    goto LABEL_9;
  if ( !result || result == -4096 )
  {
    a9[3] = -8192;
LABEL_9:
    a9[4] = 0;
    goto LABEL_10;
  }
  sub_BD60C0(a9 + 1);
  a9[3] = v13;
  result = v14;
  a9[4] = v14;
LABEL_10:
  --*(_DWORD *)(a1 + 16);
  ++*(_DWORD *)(a1 + 20);
  return result;
}
