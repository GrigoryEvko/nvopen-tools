// Function: sub_37E1980
// Address: 0x37e1980
//
__int64 __fastcall sub_37E1980(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        unsigned __int8 (__fastcall *a4)(__int64 *, __int64 *))
{
  __int64 result; // rax
  unsigned __int64 v6; // r15
  unsigned __int64 v7; // rbx
  __int64 *v8; // r14
  __int64 v9; // rax
  int v10; // ecx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // [rsp+10h] [rbp-40h]
  unsigned __int64 v14; // [rsp+18h] [rbp-38h]

  result = (__int64)a2 - a1;
  v14 = (unsigned __int64)a2;
  v13 = a3;
  if ( (__int64)a2 - a1 <= 256 )
    return result;
  if ( !a3 )
  {
    v8 = a2;
    goto LABEL_13;
  }
  while ( 2 )
  {
    v6 = v14;
    v7 = a1 + 16;
    --v13;
    sub_37DD360((__int64 *)a1, (__int64 *)(a1 + 16), (__int64 *)(a1 + 16 * (result >> 5)), (__int64 *)(v14 - 16), a4);
    while ( 1 )
    {
      v8 = (__int64 *)v7;
      if ( a4((__int64 *)v7, (__int64 *)a1) )
        goto LABEL_4;
      do
        v6 -= 16LL;
      while ( a4((__int64 *)a1, (__int64 *)v6) );
      if ( v7 >= v6 )
        break;
      v9 = *(_QWORD *)v7;
      *(_QWORD *)v7 = *(_QWORD *)v6;
      v10 = *(_DWORD *)(v6 + 8);
      *(_QWORD *)v6 = v9;
      LODWORD(v9) = *(_DWORD *)(v7 + 8);
      *(_DWORD *)(v7 + 8) = v10;
      *(_DWORD *)(v6 + 8) = v9;
LABEL_4:
      v7 += 16LL;
    }
    sub_37E1980(v7, v14, v13, a4);
    result = v7 - a1;
    if ( (__int64)(v7 - a1) > 256 )
    {
      if ( v13 )
      {
        v14 = v7;
        continue;
      }
LABEL_13:
      sub_37E1890(a1, v8, (unsigned __int64)v8, (unsigned __int8 (__fastcall *)(__int64, __int64))a4);
      do
      {
        v8 -= 2;
        v11 = *v8;
        v12 = v8[1];
        *v8 = *(_QWORD *)a1;
        *((_DWORD *)v8 + 2) = *(_DWORD *)(a1 + 8);
        result = sub_37E1680(
                   a1,
                   0,
                   ((__int64)v8 - a1) >> 4,
                   v11,
                   v12,
                   (unsigned __int8 (__fastcall *)(__int64, __int64))a4);
      }
      while ( (__int64)v8 - a1 > 16 );
    }
    return result;
  }
}
