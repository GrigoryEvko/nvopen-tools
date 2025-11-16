// Function: sub_E6C380
// Address: 0xe6c380
//
__int64 __fastcall sub_E6C380(__int64 a1, __int64 *a2, char a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbp
  __int64 v7; // rax
  __int64 v8; // r9
  __int64 v9; // rcx
  char v10; // al
  __int64 v11; // [rsp-38h] [rbp-38h] BYREF
  __int64 v12; // [rsp-30h] [rbp-30h]
  __int64 *v13; // [rsp-28h] [rbp-28h]
  __int64 v14; // [rsp-20h] [rbp-20h]
  __int16 v15; // [rsp-18h] [rbp-18h]
  __int64 v16; // [rsp-8h] [rbp-8h]

  if ( !*(_BYTE *)(a1 + 1908) )
    return sub_E6BCB0((_DWORD *)a1, 0, 1u);
  v16 = v5;
  v7 = *(_QWORD *)(a1 + 152);
  v8 = *(_QWORD *)(v7 + 88);
  v9 = *(_QWORD *)(v7 + 96);
  v10 = *((_BYTE *)a2 + 32);
  if ( v10 )
  {
    if ( v10 == 1 )
    {
      v11 = v8;
      v12 = v9;
      v15 = 261;
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
      v11 = v8;
      v12 = v9;
      v13 = a2;
      v14 = a5;
      LOBYTE(v15) = 5;
      HIBYTE(v15) = v10;
    }
  }
  else
  {
    v15 = 256;
  }
  return sub_E6BFC0((_DWORD *)a1, (__int64)&v11, a3, 1u);
}
