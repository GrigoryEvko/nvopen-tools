// Function: sub_1584C50
// Address: 0x1584c50
//
__int64 __fastcall sub_1584C50(__int64 *a1, __int64 a2, _DWORD *a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v6; // rdi
  int v8; // r15d
  unsigned int v9; // r13d
  __int64 v10; // r8
  __int64 v11; // rdx
  __int64 v12; // rdx
  _BYTE *v13; // rsi
  char v14; // al
  __int64 v15; // [rsp+0h] [rbp-160h]
  _DWORD *v16; // [rsp+8h] [rbp-158h]
  __int64 v18; // [rsp+18h] [rbp-148h]
  _BYTE *v19; // [rsp+20h] [rbp-140h] BYREF
  __int64 v20; // [rsp+28h] [rbp-138h]
  _BYTE v21[304]; // [rsp+30h] [rbp-130h] BYREF

  result = a2;
  if ( a4 )
  {
    v6 = *a1;
    if ( *(_BYTE *)(v6 + 8) == 13 )
      v8 = *(_DWORD *)(v6 + 12);
    else
      v8 = *(_DWORD *)(v6 + 32);
    v19 = v21;
    v20 = 0x2000000000LL;
    if ( v8 )
    {
      v9 = 0;
      v16 = a3 + 1;
      while ( 1 )
      {
        result = sub_15A0A60(a1, v9);
        if ( !result )
          goto LABEL_13;
        if ( *a3 == v9 )
        {
          result = sub_1584C50(result, a2, v16, a4 - 1, v10);
          v11 = (unsigned int)v20;
          if ( (unsigned int)v20 >= HIDWORD(v20) )
          {
LABEL_18:
            v15 = result;
            sub_16CD150(&v19, v21, 0, 8);
            v11 = (unsigned int)v20;
            result = v15;
          }
        }
        else
        {
          v11 = (unsigned int)v20;
          if ( (unsigned int)v20 >= HIDWORD(v20) )
            goto LABEL_18;
        }
        ++v9;
        *(_QWORD *)&v19[8 * v11] = result;
        v12 = (unsigned int)(v20 + 1);
        LODWORD(v20) = v20 + 1;
        if ( v8 == v9 )
        {
          v6 = *a1;
          v13 = v19;
          goto LABEL_11;
        }
      }
    }
    v13 = v21;
    v12 = 0;
LABEL_11:
    v14 = *(_BYTE *)(v6 + 8);
    if ( v14 == 13 )
    {
      result = sub_159F090(v6, v13, v12);
    }
    else if ( v14 == 14 )
    {
      result = sub_159DFD0(v6, v13, v12);
    }
    else
    {
      result = sub_15A01B0(v13, v12);
    }
LABEL_13:
    if ( v19 != v21 )
    {
      v18 = result;
      _libc_free((unsigned __int64)v19);
      return v18;
    }
  }
  return result;
}
