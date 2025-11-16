// Function: sub_14D5D40
// Address: 0x14d5d40
//
__int64 __fastcall sub_14D5D40(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // r12
  __int64 result; // rax
  unsigned int v7; // eax
  __int64 v8; // rcx
  char v9; // dl
  __int64 v11; // [rsp+8h] [rbp-38h]
  unsigned __int8 v12; // [rsp+8h] [rbp-38h]
  unsigned __int8 v13; // [rsp+8h] [rbp-38h]
  __int64 v14; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-28h]

  v5 = (_QWORD *)a1;
  if ( *(_BYTE *)(a1 + 16) > 3u )
  {
    *a2 = 0;
    if ( *(_BYTE *)(a1 + 16) != 5 )
      return 0;
    while ( (*((_WORD *)v5 + 9) & 0xFFFD) == 0x2D )
    {
      v5 = (_QWORD *)v5[-3 * (*((_DWORD *)v5 + 5) & 0xFFFFFFF)];
      if ( *((_BYTE *)v5 + 16) <= 3u )
        goto LABEL_2;
      *a2 = 0;
      if ( *((_BYTE *)v5 + 16) != 5 )
        return 0;
    }
    if ( *((_WORD *)v5 + 9) == 32 )
    {
      v7 = sub_15A95F0(a4, *v5);
      v8 = a4;
      v15 = v7;
      if ( v7 > 0x40 )
      {
        sub_16A4EF0(&v14, 0, 0);
        v8 = a4;
      }
      else
      {
        v14 = 0;
      }
      v11 = v8;
      if ( (unsigned __int8)sub_14D5D40(v5[-3 * (*((_DWORD *)v5 + 5) & 0xFFFFFFF)], a2, &v14, v8)
        && (result = sub_1634900(v5, v11, &v14), (_BYTE)result) )
      {
        if ( *(_DWORD *)(a3 + 8) <= 0x40u )
        {
          v9 = v15;
          if ( v15 <= 0x40 )
          {
            *(_DWORD *)(a3 + 8) = v15;
            *(_QWORD *)a3 = v14 & (0xFFFFFFFFFFFFFFFFLL >> -v9);
            return result;
          }
        }
        v12 = result;
        sub_16A51C0(a3, &v14);
        result = v12;
      }
      else
      {
        result = 0;
      }
      if ( v15 > 0x40 && v14 )
      {
        v13 = result;
        j_j___libc_free_0_0(v14);
        return v13;
      }
    }
    else
    {
      return 0;
    }
  }
  else
  {
LABEL_2:
    *a2 = v5;
    v15 = sub_15A95F0(a4, *v5);
    if ( v15 <= 0x40 )
      v14 = 0;
    else
      sub_16A4EF0(&v14, 0, 0);
    if ( *(_DWORD *)(a3 + 8) > 0x40u )
    {
      if ( *(_QWORD *)a3 )
        j_j___libc_free_0_0(*(_QWORD *)a3);
    }
    *(_QWORD *)a3 = v14;
    *(_DWORD *)(a3 + 8) = v15;
    return 1;
  }
  return result;
}
