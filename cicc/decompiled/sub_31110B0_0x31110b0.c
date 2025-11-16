// Function: sub_31110B0
// Address: 0x31110b0
//
_QWORD *__fastcall sub_31110B0(_QWORD *a1, int *a2, __int64 a3)
{
  int v4; // eax
  char v5; // bl
  _QWORD *v6; // rsi
  _QWORD *v7; // rdx
  int v9; // [rsp+0h] [rbp-60h] BYREF
  const char *v10; // [rsp+8h] [rbp-58h]
  __int64 v11; // [rsp+10h] [rbp-50h]
  int v12; // [rsp+18h] [rbp-48h]
  __int64 v13; // [rsp+20h] [rbp-40h]
  __int64 v14; // [rsp+28h] [rbp-38h]
  __int64 v15; // [rsp+30h] [rbp-30h]

  v4 = *a2;
  v9 = 0;
  v10 = 0;
  v11 = 0;
  v12 = v4;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  if ( v4 )
  {
    if ( v4 == 1 )
    {
      v11 = 27;
      v10 = "__guard_dispatch_icall_fptr";
    }
  }
  else
  {
    v11 = 24;
    v10 = "__guard_check_icall_fptr";
  }
  v5 = sub_310F9A0((__int64)&v9, *(__int64 ***)(a3 + 40));
  if ( v9 == 2 )
    v5 |= sub_310FBD0((__int64)&v9, a3);
  v6 = a1 + 4;
  v7 = a1 + 10;
  if ( v5 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v6;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v7;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
    a1[1] = v6;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[4] = &qword_4F82400;
    a1[7] = v7;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    *a1 = 1;
  }
  return a1;
}
