// Function: sub_E56D00
// Address: 0xe56d00
//
__int64 __fastcall sub_E56D00(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbp
  char v5; // al
  __int64 v6; // rdi
  _QWORD v8[4]; // [rsp-38h] [rbp-38h] BYREF
  __int16 v9; // [rsp-18h] [rbp-18h]
  __int64 v10; // [rsp-8h] [rbp-8h]

  if ( !*(_BYTE *)(*(_QWORD *)(a1 + 312) + 21LL) )
    return sub_E9A190();
  v10 = v4;
  v5 = *((_BYTE *)a2 + 32);
  v6 = *(_QWORD *)(a1 + 8);
  if ( v5 )
  {
    if ( v5 == 1 )
    {
      v8[0] = "_end";
      v9 = 259;
    }
    else
    {
      if ( *((_BYTE *)a2 + 33) == 1 )
      {
        a4 = a2[1];
        a2 = (__int64 *)*a2;
      }
      else
      {
        v5 = 2;
      }
      v8[1] = a4;
      v8[0] = a2;
      v8[2] = "_end";
      LOBYTE(v9) = v5;
      HIBYTE(v9) = 3;
    }
  }
  else
  {
    v9 = 256;
  }
  return sub_E6C380(v6, v8, 1);
}
