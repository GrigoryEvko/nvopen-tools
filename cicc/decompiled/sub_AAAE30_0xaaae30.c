// Function: sub_AAAE30
// Address: 0xaaae30
//
__int64 __fastcall sub_AAAE30(__int64 a1, __int64 a2, _DWORD *a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v7; // rdi
  int v9; // r15d
  unsigned int v10; // r13d
  unsigned __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // [rsp+0h] [rbp-160h]
  _DWORD *v15; // [rsp+8h] [rbp-158h]
  __int64 v17; // [rsp+18h] [rbp-148h]
  _BYTE *v18; // [rsp+20h] [rbp-140h] BYREF
  __int64 v19; // [rsp+28h] [rbp-138h]
  _BYTE v20[304]; // [rsp+30h] [rbp-130h] BYREF

  result = a2;
  if ( a4 )
  {
    v7 = *(_QWORD *)(a1 + 8);
    if ( *(_BYTE *)(v7 + 8) == 15 )
      v9 = *(_DWORD *)(v7 + 12);
    else
      v9 = *(_DWORD *)(v7 + 32);
    v18 = v20;
    v19 = 0x2000000000LL;
    if ( v9 )
    {
      v10 = 0;
      v15 = a3 + 1;
      while ( 1 )
      {
        v11 = v10;
        result = sub_AD69F0(a1, v10);
        if ( !result )
          break;
        if ( *a3 == v10 )
          result = sub_AAAE30(result, a2, v15, a4 - 1);
        v12 = (unsigned int)v19;
        if ( (unsigned __int64)(unsigned int)v19 + 1 > HIDWORD(v19) )
        {
          v14 = result;
          sub_C8D5F0(&v18, v20, (unsigned int)v19 + 1LL, 8);
          v12 = (unsigned int)v19;
          result = v14;
        }
        ++v10;
        *(_QWORD *)&v18[8 * v12] = result;
        v13 = (unsigned int)(v19 + 1);
        LODWORD(v19) = v19 + 1;
        if ( v9 == v10 )
        {
          v7 = *(_QWORD *)(a1 + 8);
          v11 = (unsigned __int64)v18;
          goto LABEL_13;
        }
      }
    }
    else
    {
      v11 = (unsigned __int64)v20;
      v13 = 0;
LABEL_13:
      if ( *(_BYTE *)(v7 + 8) == 15 )
        result = sub_AD24A0(v7, v11, v13);
      else
        result = sub_AD1300(v7, v11, v13);
    }
    if ( v18 != v20 )
    {
      v17 = result;
      _libc_free(v18, v11);
      return v17;
    }
  }
  return result;
}
