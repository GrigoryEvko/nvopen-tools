// Function: sub_920540
// Address: 0x920540
//
__int64 __fastcall sub_920540(__int64 a1, __int64 a2, _BYTE *a3, _BYTE **a4, __int64 a5, int a6)
{
  _BYTE **v10; // rax
  _BYTE **v11; // rcx
  __int64 result; // rax
  __int64 v13; // [rsp+8h] [rbp-68h]
  __int64 v14; // [rsp+8h] [rbp-68h]
  __int64 v15; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v16; // [rsp+18h] [rbp-58h]
  __int64 v17; // [rsp+20h] [rbp-50h]
  unsigned int v18; // [rsp+28h] [rbp-48h]
  char v19; // [rsp+30h] [rbp-40h]

  if ( (unsigned __int8)sub_BCEA30(a2) )
    return 0;
  if ( *a3 > 0x15u )
    return 0;
  v10 = sub_920370(a4, (__int64)&a4[a5]);
  if ( v11 != v10 )
    return 0;
  v19 = 0;
  result = sub_AD9FD0(a2, (_DWORD)a3, (_DWORD)a4, a5, a6, (unsigned int)&v15, 0);
  if ( v19 )
  {
    v19 = 0;
    if ( v18 > 0x40 && v17 )
    {
      v13 = result;
      j_j___libc_free_0_0(v17);
      result = v13;
    }
    if ( v16 > 0x40 )
    {
      if ( v15 )
      {
        v14 = result;
        j_j___libc_free_0_0(v15);
        return v14;
      }
    }
  }
  return result;
}
