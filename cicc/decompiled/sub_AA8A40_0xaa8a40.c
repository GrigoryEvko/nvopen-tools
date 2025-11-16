// Function: sub_AA8A40
// Address: 0xaa8a40
//
__int64 __fastcall sub_AA8A40(__int64 *a1, __int64 *a2)
{
  __int64 v3; // rdx
  unsigned int v4; // r8d
  char v5; // al
  __int64 result; // rax
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned int v10; // [rsp+Ch] [rbp-34h]
  __int64 v11; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-28h]
  char v13; // [rsp+1Ch] [rbp-24h]
  __int64 v14; // [rsp+20h] [rbp-20h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-18h]

  v3 = *((unsigned int *)a1 + 2);
  v4 = *((_DWORD *)a2 + 2);
  if ( v4 == (_DWORD)v3 )
  {
    v5 = *((_BYTE *)a1 + 12);
    if ( v5 == *((_BYTE *)a2 + 12) )
    {
      if ( !v5 )
        return sub_C4C880(a1, a2);
      return sub_C49970(a1, a2);
    }
LABEL_15:
    if ( v5 )
    {
      if ( !*((_BYTE *)a2 + 12) )
      {
        v9 = *a2;
        if ( v4 > 0x40 )
          v9 = *(_QWORD *)(v9 + 8LL * ((v4 - 1) >> 6));
        if ( (v9 & (1LL << ((unsigned __int8)v4 - 1))) != 0 )
          return 1;
      }
    }
    else
    {
      v7 = 1LL << ((unsigned __int8)v3 - 1);
      v8 = *a1;
      if ( (unsigned int)v3 > 0x40 )
      {
        if ( (*(_QWORD *)(v8 + 8LL * ((unsigned int)(v3 - 1) >> 6)) & v7) != 0 )
          return 0xFFFFFFFFLL;
      }
      else if ( (v8 & v7) != 0 )
      {
        return 0xFFFFFFFFLL;
      }
    }
    return sub_C49970(a1, a2);
  }
  if ( v4 >= (unsigned int)v3 )
  {
    v5 = *((_BYTE *)a1 + 12);
    if ( v4 > (unsigned int)v3 )
    {
      if ( v5 )
      {
        sub_C449B0(&v14, a1, v4);
        v13 = 1;
      }
      else
      {
        sub_C44830(&v14, a1, v4);
        v13 = 0;
      }
      v12 = v15;
      v11 = v14;
      result = sub_AA8A40(&v11, a2);
      if ( v12 <= 0x40 )
        return result;
LABEL_11:
      if ( v11 )
      {
        v10 = result;
        j_j___libc_free_0_0(v11);
        return v10;
      }
      return result;
    }
    goto LABEL_15;
  }
  if ( *((_BYTE *)a2 + 12) )
  {
    sub_C449B0(&v14, a2, v3);
    v13 = 1;
  }
  else
  {
    sub_C44830(&v14, a2, v3);
    v13 = 0;
  }
  v12 = v15;
  v11 = v14;
  result = sub_AA8A40(a1, &v11);
  if ( v12 > 0x40 )
    goto LABEL_11;
  return result;
}
