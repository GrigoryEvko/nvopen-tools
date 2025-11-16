// Function: sub_E6D8A0
// Address: 0xe6d8a0
//
__int64 __fastcall sub_E6D8A0(__int64 a1, void **a2, char a3, int a4, __int64 a5, int a6)
{
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // rax
  int v13; // [rsp+4h] [rbp-5Ch]
  int v14; // [rsp+8h] [rbp-58h]
  __int64 v15; // [rsp+8h] [rbp-58h]
  int v16; // [rsp+8h] [rbp-58h]
  __int64 *v17; // [rsp+10h] [rbp-50h] BYREF
  __int64 v18; // [rsp+18h] [rbp-48h]
  __int64 v19; // [rsp+20h] [rbp-40h] BYREF

  if ( *(_BYTE *)(a5 + 32) <= 1u )
  {
    v10 = 0;
  }
  else
  {
    v14 = a6;
    sub_CA0F50((__int64 *)&v17, (void **)a5);
    v9 = v18;
    a6 = v14;
    if ( v17 != &v19 )
    {
      v15 = v18;
      v13 = a6;
      j_j___libc_free_0(v17, v19 + 1);
      a6 = v13;
      v9 = v15;
    }
    v10 = 0;
    if ( v9 )
    {
      v16 = a6;
      v11 = sub_E6C460(a1, (const char **)a5);
      a6 = v16;
      *(_BYTE *)(v11 + 42) = 1;
      v10 = v11;
      if ( !a3 && !*(_BYTE *)(v11 + 36) )
      {
        *(_DWORD *)(v11 + 32) = 3;
        *(_BYTE *)(v11 + 36) = 1;
      }
    }
  }
  return sub_E6D520(a1, a2, a3, a4, v10, a6);
}
