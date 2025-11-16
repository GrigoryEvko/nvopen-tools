// Function: sub_E71CB0
// Address: 0xe71cb0
//
unsigned __int64 __fastcall sub_E71CB0(
        __int64 a1,
        size_t *a2,
        int a3,
        unsigned int a4,
        int a5,
        __int64 a6,
        unsigned __int8 a7,
        int a8,
        __int64 a9)
{
  __int64 v10; // rax
  __int64 v11; // r9
  __int64 v12; // rax
  int v14; // [rsp+8h] [rbp-68h]
  int v15; // [rsp+Ch] [rbp-64h]
  unsigned int v16; // [rsp+Ch] [rbp-64h]
  int v17; // [rsp+Ch] [rbp-64h]
  int v18; // [rsp+10h] [rbp-60h]
  __int64 v19; // [rsp+10h] [rbp-60h]
  int v20; // [rsp+10h] [rbp-60h]
  unsigned int v21; // [rsp+18h] [rbp-58h]
  int v22; // [rsp+18h] [rbp-58h]
  unsigned int v23; // [rsp+18h] [rbp-58h]
  __int64 *v24; // [rsp+20h] [rbp-50h] BYREF
  __int64 v25; // [rsp+28h] [rbp-48h]
  __int64 v26; // [rsp+30h] [rbp-40h] BYREF

  if ( *(_BYTE *)(a6 + 32) <= 1u )
  {
    v11 = 0;
  }
  else
  {
    v15 = a5;
    v21 = a4;
    v18 = a3;
    sub_CA0F50((__int64 *)&v24, (void **)a6);
    a3 = v18;
    v10 = v25;
    a4 = v21;
    a5 = v15;
    if ( v24 != &v26 )
    {
      v19 = v25;
      v14 = v15;
      v16 = v21;
      v22 = a3;
      j_j___libc_free_0(v24, v26 + 1);
      a5 = v14;
      a4 = v16;
      a3 = v22;
      v10 = v19;
    }
    v11 = 0;
    if ( v10 )
    {
      v17 = a5;
      v23 = a4;
      v20 = a3;
      v12 = sub_E6C460(a1, (const char **)a6);
      a5 = v17;
      a4 = v23;
      a3 = v20;
      v11 = v12;
    }
  }
  return sub_E713E0(a1, a2, a3, a4, a5, v11, a7, a8, a9);
}
