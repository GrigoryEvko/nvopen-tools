// Function: sub_2A624B0
// Address: 0x2a624b0
//
bool __fastcall sub_2A624B0(__int64 a1, unsigned __int8 *a2, char a3)
{
  char v4; // di
  int v5; // ecx
  bool result; // al
  bool v8; // [rsp+Fh] [rbp-51h]
  bool v9; // [rsp+Fh] [rbp-51h]
  bool v10; // [rsp+Fh] [rbp-51h]
  unsigned __int64 v11; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-48h]
  unsigned __int64 v13; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v14; // [rsp+28h] [rbp-38h]
  unsigned __int64 v15; // [rsp+30h] [rbp-30h]
  unsigned int v16; // [rsp+38h] [rbp-28h]

  v4 = *(_BYTE *)a1;
  v5 = *a2;
  result = 0;
  if ( (unsigned int)(v5 - 12) > 1 )
  {
    if ( v4 != 2 )
    {
      if ( (_BYTE)v5 == 17 )
      {
        v12 = *((_DWORD *)a2 + 8);
        if ( v12 > 0x40 )
          sub_C43780((__int64)&v11, (const void **)a2 + 3);
        else
          v11 = *((_QWORD *)a2 + 3);
        sub_AADBC0((__int64)&v13, (__int64 *)&v11);
        result = sub_2A62120((char *)a1, (__int64)&v13, a3, 0, 1u);
        if ( v16 > 0x40 && v15 )
        {
          v8 = result;
          j_j___libc_free_0_0(v15);
          result = v8;
        }
        if ( v14 > 0x40 && v13 )
        {
          v9 = result;
          j_j___libc_free_0_0(v13);
          result = v9;
        }
        if ( v12 > 0x40 && v11 )
        {
          v10 = result;
          j_j___libc_free_0_0(v11);
          return v10;
        }
      }
      else
      {
        *(_BYTE *)a1 = 2;
        *(_QWORD *)(a1 + 8) = a2;
        return 1;
      }
    }
  }
  else if ( v4 != 1 )
  {
    *(_BYTE *)a1 = 1;
    return 1;
  }
  return result;
}
