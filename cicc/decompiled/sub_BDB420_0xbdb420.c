// Function: sub_BDB420
// Address: 0xbdb420
//
unsigned __int64 __fastcall sub_BDB420(__int64 a1, const char *a2, const char *a3)
{
  unsigned __int8 v6; // dl
  __int64 v7; // rcx
  unsigned __int64 result; // rax
  unsigned __int64 v9; // rcx
  const char *v10; // r15
  __int64 v11; // rbx
  _BYTE *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdi
  _BYTE *v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // r15
  _BYTE *v19; // rax
  char v20; // dl
  _QWORD v21[4]; // [rsp+0h] [rbp-60h] BYREF
  char v22; // [rsp+20h] [rbp-40h]
  char v23; // [rsp+21h] [rbp-3Fh]

  if ( *a3 == 5 )
  {
    v6 = *(a3 - 16);
    if ( (v6 & 2) != 0 )
    {
      result = *((_QWORD *)a3 - 4);
      v7 = *((unsigned int *)a3 - 6);
    }
    else
    {
      v7 = (*((_WORD *)a3 - 8) >> 6) & 0xF;
      result = (unsigned __int64)&a3[-8 * ((v6 >> 2) & 0xF) - 16];
    }
    v9 = result + 8 * v7;
    if ( result == v9 )
      return result;
    while ( 1 )
    {
      v10 = *(const char **)result;
      if ( !*(_QWORD *)result || (unsigned int)*(unsigned __int8 *)v10 - 23 > 1 )
        break;
      result += 8LL;
      if ( v9 == result )
        return result;
    }
    v11 = *(_QWORD *)a1;
    v23 = 1;
    v21[0] = "invalid template parameter";
    v22 = 3;
    if ( v11 )
    {
      sub_CA0E80(v21, v11);
      v12 = *(_BYTE **)(v11 + 32);
      if ( (unsigned __int64)v12 >= *(_QWORD *)(v11 + 24) )
      {
        sub_CB5D20(v11, 10);
      }
      else
      {
        *(_QWORD *)(v11 + 32) = v12 + 1;
        *v12 = 10;
      }
      v13 = *(_QWORD *)a1;
      result = *(unsigned __int8 *)(a1 + 154);
      *(_BYTE *)(a1 + 153) = 1;
      *(_BYTE *)(a1 + 152) |= result;
      if ( v13 )
      {
        sub_A62C00(a2, v13, a1 + 16, *(_QWORD *)(a1 + 8));
        v14 = *(_QWORD *)a1;
        v15 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
        if ( (unsigned __int64)v15 >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
        {
          sub_CB5D20(v14, 10);
        }
        else
        {
          *(_QWORD *)(v14 + 32) = v15 + 1;
          *v15 = 10;
        }
        sub_A62C00(a3, *(_QWORD *)a1, a1 + 16, *(_QWORD *)(a1 + 8));
        v16 = *(_QWORD *)a1;
        result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
        if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
        {
          result = sub_CB5D20(v16, 10);
        }
        else
        {
          *(_QWORD *)(v16 + 32) = result + 1;
          *(_BYTE *)result = 10;
        }
        if ( v10 )
        {
          sub_A62C00(v10, *(_QWORD *)a1, a1 + 16, *(_QWORD *)(a1 + 8));
          v17 = *(_QWORD *)a1;
          result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
          if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
          {
            return sub_CB5D20(v17, 10);
          }
          else
          {
            *(_QWORD *)(v17 + 32) = result + 1;
            *(_BYTE *)result = 10;
          }
        }
      }
      return result;
    }
LABEL_21:
    result = *(unsigned __int8 *)(a1 + 154);
    *(_BYTE *)(a1 + 153) = 1;
    *(_BYTE *)(a1 + 152) |= result;
    return result;
  }
  v18 = *(_QWORD *)a1;
  v23 = 1;
  v21[0] = "invalid template params";
  v22 = 3;
  if ( !v18 )
    goto LABEL_21;
  sub_CA0E80(v21, v18);
  v19 = *(_BYTE **)(v18 + 32);
  if ( (unsigned __int64)v19 >= *(_QWORD *)(v18 + 24) )
  {
    sub_CB5D20(v18, 10);
  }
  else
  {
    *(_QWORD *)(v18 + 32) = v19 + 1;
    *v19 = 10;
  }
  result = *(_QWORD *)a1;
  v20 = *(_BYTE *)(a1 + 154);
  *(_BYTE *)(a1 + 153) = 1;
  *(_BYTE *)(a1 + 152) |= v20;
  if ( result )
  {
    sub_BD9900((__int64 *)a1, a2);
    return (unsigned __int64)sub_BD9900((__int64 *)a1, a3);
  }
  return result;
}
