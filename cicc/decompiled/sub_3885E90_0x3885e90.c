// Function: sub_3885E90
// Address: 0x3885e90
//
void __fastcall sub_3885E90(__int64 a1, _BYTE *a2, _BYTE *a3, __int64 *a4)
{
  __int64 v4; // rbp
  __int64 v8; // rax
  int v9; // r8d
  __int64 v10; // rdx
  __int64 v11; // rax
  int v12; // ecx
  __int64 v13; // rax
  int v14; // r8d
  __int64 v15; // rdx
  __int64 v16; // rax
  int v17; // ecx
  unsigned __int64 v18; // rsi
  const char *v19; // [rsp-38h] [rbp-38h] BYREF
  char v20; // [rsp-28h] [rbp-28h]
  char v21; // [rsp-27h] [rbp-27h]
  __int64 v22; // [rsp-8h] [rbp-8h]

  a4[1] = 0;
  if ( a2 == a3 )
  {
    *a4 = 0;
  }
  else
  {
    v22 = v4;
    v8 = 0;
    v9 = 0;
    do
    {
      v11 = 16 * v8;
      a4[1] = v11;
      v12 = (char)*a2;
      if ( (unsigned __int8)(*a2 - 48) > 9u )
      {
        if ( (unsigned __int8)(v12 - 97) <= 5u )
        {
          v10 = (unsigned int)(v12 - 87);
        }
        else
        {
          v10 = 0xFFFFFFFFLL;
          if ( (unsigned __int8)(v12 - 65) <= 5u )
            v10 = (unsigned int)(v12 - 55);
        }
      }
      else
      {
        v10 = (unsigned int)(char)(v12 - 48);
      }
      v8 = v10 + v11;
      ++v9;
      ++a2;
      a4[1] = v8;
    }
    while ( v9 <= 3 && a3 != a2 );
    *a4 = 0;
    if ( a3 != a2 )
    {
      v13 = 0;
      v14 = 0;
      while ( 1 )
      {
        v16 = 16 * v13;
        *a4 = v16;
        v17 = (char)*a2;
        if ( (unsigned __int8)(*a2 - 48) > 9u )
        {
          if ( (unsigned __int8)(v17 - 97) <= 5u )
          {
            v15 = (unsigned int)(v17 - 87);
          }
          else
          {
            v15 = 0xFFFFFFFFLL;
            if ( (unsigned __int8)(v17 - 65) <= 5u )
              v15 = (unsigned int)(v17 - 55);
          }
        }
        else
        {
          v15 = (unsigned int)(char)(v17 - 48);
        }
        v13 = v15 + v16;
        ++v14;
        ++a2;
        *a4 = v13;
        if ( v14 > 15 )
          break;
        if ( a3 == a2 )
          return;
      }
      if ( a3 != a2 )
      {
        v18 = *(_QWORD *)(a1 + 48);
        v21 = 1;
        v20 = 3;
        v19 = "constant bigger than 128 bits detected!";
        sub_38814C0(a1, v18, (__int64)&v19);
      }
    }
  }
}
