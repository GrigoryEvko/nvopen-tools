// Function: sub_2916060
// Address: 0x2916060
//
__int64 __fastcall sub_2916060(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  __int64 result; // rax
  unsigned __int8 *v4; // r12
  int v8; // edx
  unsigned __int8 *v9; // rdx
  unsigned __int8 *v10; // r13
  __int64 v11; // r13
  unsigned __int8 *i; // rdx
  _BYTE *v13; // rdi
  unsigned int v14; // esi
  __int64 v15; // rax
  unsigned __int8 **v16; // rdi
  unsigned __int8 **v17; // rdx
  unsigned __int8 **v18; // rax
  __int64 v19; // rsi
  __int64 *v20; // rax
  unsigned __int8 *v21; // [rsp+8h] [rbp-38h]

  result = -32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v4 = *(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  if ( a3 != v4 )
  {
    while ( 1 )
    {
      result = *((_QWORD *)v4 + 2);
      if ( result )
      {
        if ( *(_QWORD *)(result + 8) )
          return result;
        result = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
        if ( (unsigned __int8 *)result != v4 || !result )
          return result;
      }
      result = *v4;
      if ( (unsigned __int8)result <= 0x1Cu )
      {
        if ( (_BYTE)result != 5 )
          return result;
        result = *((unsigned __int16 *)v4 + 1);
        if ( (_WORD)result != 34 )
        {
          if ( (_WORD)result != 49 )
          {
            v8 = (unsigned __int16)result;
            goto LABEL_10;
          }
LABEL_11:
          if ( (v4[7] & 0x40) != 0 )
            v9 = (unsigned __int8 *)*((_QWORD *)v4 - 1);
          else
            v9 = &v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
          v10 = *(unsigned __int8 **)v9;
          goto LABEL_24;
        }
      }
      else if ( (_BYTE)result != 63 )
      {
        v8 = result - 29;
        if ( (_DWORD)result != 78 )
        {
LABEL_10:
          if ( v8 != 50 )
            return result;
        }
        goto LABEL_11;
      }
      v11 = *((_DWORD *)v4 + 1) & 0x7FFFFFF;
      for ( i = &v4[32 * (1 - v11)]; i != v4; i += 32 )
      {
        v13 = *(_BYTE **)i;
        if ( **(_BYTE **)i != 17 )
          return result;
        v14 = *((_DWORD *)v13 + 8);
        if ( v14 <= 0x40 )
        {
          if ( *((_QWORD *)v13 + 3) )
            return result;
        }
        else
        {
          v21 = i;
          result = sub_C444A0((__int64)(v13 + 24));
          i = v21;
          if ( v14 != (_DWORD)result )
            return result;
        }
      }
      v10 = *(unsigned __int8 **)&v4[-32 * v11];
LABEL_24:
      v15 = sub_ACADE0(*((__int64 ***)v4 + 1));
      sub_BD84D0((__int64)v4, v15);
      if ( *(_BYTE *)(a1 + 108) )
      {
        v16 = *(unsigned __int8 ***)(a1 + 88);
        v17 = &v16[*(unsigned int *)(a1 + 100)];
        v18 = v16;
        if ( v16 != v17 )
        {
          while ( *v18 != v4 )
          {
            if ( v17 == ++v18 )
              goto LABEL_30;
          }
          v19 = (unsigned int)(*(_DWORD *)(a1 + 100) - 1);
          *(_DWORD *)(a1 + 100) = v19;
          *v18 = v16[v19];
          ++*(_QWORD *)(a1 + 80);
        }
      }
      else
      {
        v20 = sub_C8CA60(a1 + 80, (__int64)v4);
        if ( v20 )
        {
          *v20 = -2;
          ++*(_DWORD *)(a1 + 104);
          ++*(_QWORD *)(a1 + 80);
        }
      }
LABEL_30:
      result = sub_B43D60(v4);
      if ( v10 == a3 )
        return result;
      v4 = v10;
    }
  }
  return result;
}
