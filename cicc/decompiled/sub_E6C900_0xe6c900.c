// Function: sub_E6C900
// Address: 0xe6c900
//
__int64 __fastcall sub_E6C900(__int64 a1, __int64 *a2, char a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 v6; // cl
  __int64 v7; // rax
  const char *v8; // r9
  __int64 v9; // rdx
  char v10; // al
  __int64 v12; // rax
  const char *v13; // rcx
  __int64 v14; // rdx
  char v15; // al
  const char *v16; // [rsp+0h] [rbp-30h] BYREF
  __int64 v17; // [rsp+8h] [rbp-28h]
  __int64 *v18; // [rsp+10h] [rbp-20h]
  __int64 v19; // [rsp+18h] [rbp-18h]
  __int16 v20; // [rsp+20h] [rbp-10h]

  if ( a3 )
  {
    v12 = *(_QWORD *)(a1 + 152);
    v13 = *(const char **)(v12 + 104);
    v14 = *(_QWORD *)(v12 + 112);
    v15 = *((_BYTE *)a2 + 32);
    if ( v15 )
    {
      if ( v15 == 1 )
      {
        v16 = v13;
        v17 = v14;
        v20 = 261;
      }
      else
      {
        if ( *((_BYTE *)a2 + 33) == 1 )
        {
          a6 = a2[1];
          a2 = (__int64 *)*a2;
        }
        else
        {
          v15 = 2;
        }
        v16 = v13;
        v17 = v14;
        v18 = a2;
        v19 = a6;
        LOBYTE(v20) = 5;
        HIBYTE(v20) = v15;
      }
    }
    else
    {
      v20 = 256;
    }
    return sub_E6C460(a1, &v16);
  }
  else
  {
    v6 = *(_BYTE *)(a1 + 1907) ^ 1;
    if ( v6 && !*(_BYTE *)(a1 + 1908) )
    {
      return sub_E6BCB0((_DWORD *)a1, 0, 1u);
    }
    else
    {
      v7 = *(_QWORD *)(a1 + 152);
      v8 = *(const char **)(v7 + 104);
      v9 = *(_QWORD *)(v7 + 112);
      v10 = *((_BYTE *)a2 + 32);
      if ( v10 )
      {
        if ( v10 == 1 )
        {
          v16 = v8;
          v17 = v9;
          v20 = 261;
        }
        else
        {
          if ( *((_BYTE *)a2 + 33) == 1 )
          {
            a5 = a2[1];
            a2 = (__int64 *)*a2;
          }
          else
          {
            v10 = 2;
          }
          v16 = v8;
          v17 = v9;
          v18 = a2;
          v19 = a5;
          LOBYTE(v20) = 5;
          HIBYTE(v20) = v10;
        }
      }
      else
      {
        v20 = 256;
      }
      return sub_E6BFC0((_DWORD *)a1, (__int64)&v16, 0, v6);
    }
  }
}
