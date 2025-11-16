// Function: sub_BDAE60
// Address: 0xbdae60
//
unsigned __int64 __fastcall sub_BDAE60(__int64 a1, const char *a2)
{
  unsigned __int64 result; // rax
  const char *v5; // rdx
  const char *v6; // r13
  __int64 v7; // rdx
  __int64 v8; // r14
  _BYTE *v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rdi
  _BYTE *v12; // rax
  __int64 v13; // rdi
  const char *v14; // [rsp+0h] [rbp-50h] BYREF
  char v15; // [rsp+20h] [rbp-30h]
  char v16; // [rsp+21h] [rbp-2Fh]

  result = *((unsigned __int8 *)a2 - 16);
  if ( (result & 2) != 0 )
  {
    v5 = (const char *)*((_QWORD *)a2 - 4);
  }
  else
  {
    result = 8LL * (((unsigned __int8)result >> 2) & 0xF);
    v5 = &a2[-result - 16];
  }
  v6 = (const char *)*((_QWORD *)v5 + 1);
  if ( v6 )
  {
    result = *(unsigned __int8 *)v6;
    if ( (unsigned __int8)result > 0x24u || (v7 = 0x140000F000LL, !_bittest64(&v7, result)) )
    {
      v8 = *(_QWORD *)a1;
      v16 = 1;
      v14 = "invalid type ref";
      v15 = 3;
      if ( v8 )
      {
        sub_CA0E80(&v14, v8);
        v9 = *(_BYTE **)(v8 + 32);
        if ( (unsigned __int64)v9 >= *(_QWORD *)(v8 + 24) )
        {
          sub_CB5D20(v8, 10);
        }
        else
        {
          *(_QWORD *)(v8 + 32) = v9 + 1;
          *v9 = 10;
        }
        v10 = *(_QWORD *)a1;
        result = *(unsigned __int8 *)(a1 + 154);
        *(_BYTE *)(a1 + 153) = 1;
        *(_BYTE *)(a1 + 152) |= result;
        if ( v10 )
        {
          sub_A62C00(a2, v10, a1 + 16, *(_QWORD *)(a1 + 8));
          v11 = *(_QWORD *)a1;
          v12 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
          if ( (unsigned __int64)v12 >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
          {
            sub_CB5D20(v11, 10);
          }
          else
          {
            *(_QWORD *)(v11 + 32) = v12 + 1;
            *v12 = 10;
          }
          sub_A62C00(v6, *(_QWORD *)a1, a1 + 16, *(_QWORD *)(a1 + 8));
          v13 = *(_QWORD *)a1;
          result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
          if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
          {
            return sub_CB5D20(v13, 10);
          }
          else
          {
            *(_QWORD *)(v13 + 32) = result + 1;
            *(_BYTE *)result = 10;
          }
        }
      }
      else
      {
        result = *(unsigned __int8 *)(a1 + 154);
        *(_BYTE *)(a1 + 153) = 1;
        *(_BYTE *)(a1 + 152) |= result;
      }
    }
  }
  return result;
}
