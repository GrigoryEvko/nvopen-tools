// Function: sub_29DA230
// Address: 0x29da230
//
__int64 __fastcall sub_29DA230(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 result; // rax
  __int64 v6; // rbx
  unsigned int *v7; // rdx
  unsigned int *v8; // rax
  __int64 v9; // r14
  __int64 v10; // r15
  unsigned int v11; // [rsp+0h] [rbp-C0h]
  unsigned int v12; // [rsp+8h] [rbp-B8h]
  _BYTE *v13; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v14; // [rsp+18h] [rbp-A8h]
  _BYTE v15[48]; // [rsp+20h] [rbp-A0h] BYREF
  _BYTE *v16; // [rsp+50h] [rbp-70h] BYREF
  __int64 v17; // [rsp+58h] [rbp-68h]
  _BYTE v18[96]; // [rsp+60h] [rbp-60h] BYREF

  v13 = v15;
  v14 = 0x300000000LL;
  v17 = 0x300000000LL;
  v16 = v18;
  sub_B9A9D0(a2, (__int64)&v13);
  sub_B9A9D0(a3, (__int64)&v16);
  v4 = (unsigned int)v14;
  result = 1;
  if ( (unsigned int)v14 <= (unsigned __int64)(unsigned int)v17 )
  {
    if ( (unsigned int)v14 < (unsigned __int64)(unsigned int)v17 )
    {
      result = 0xFFFFFFFFLL;
    }
    else
    {
      v6 = 0;
      if ( (_DWORD)v14 )
      {
        while ( 1 )
        {
          v7 = (unsigned int *)&v13[16 * v6];
          v8 = (unsigned int *)&v16[16 * v6];
          v9 = *((_QWORD *)v7 + 1);
          v10 = *((_QWORD *)v8 + 1);
          result = sub_29D7CF0((__int64)a1, *v7, *v8);
          if ( (_DWORD)result )
            break;
          result = sub_29DA080(a1, v9, v10);
          if ( (_DWORD)result )
            break;
          if ( v4 == ++v6 )
            goto LABEL_13;
        }
      }
      else
      {
LABEL_13:
        result = 0;
      }
    }
  }
  if ( v16 != v18 )
  {
    v11 = result;
    _libc_free((unsigned __int64)v16);
    result = v11;
  }
  if ( v13 != v15 )
  {
    v12 = result;
    _libc_free((unsigned __int64)v13);
    return v12;
  }
  return result;
}
