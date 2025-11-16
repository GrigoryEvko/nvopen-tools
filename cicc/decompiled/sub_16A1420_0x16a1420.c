// Function: sub_16A1420
// Address: 0x16a1420
//
__int64 __fastcall sub_16A1420(__int64 a1, __int64 a2, unsigned int a3, double a4, double a5, double a6)
{
  void *v9; // rax
  void *v10; // r13
  __int64 result; // rax
  _BYTE *v12; // rdi
  char *v13; // rax
  char v14; // al
  __int64 *v15; // rsi
  __int64 v16; // r12
  __int64 v17; // rbx
  unsigned int v18; // [rsp+Ch] [rbp-54h]
  int v19; // [rsp+Ch] [rbp-54h]
  unsigned int v20; // [rsp+Ch] [rbp-54h]
  char v21[8]; // [rsp+10h] [rbp-50h] BYREF
  void *v22; // [rsp+18h] [rbp-48h] BYREF
  __int64 v23; // [rsp+20h] [rbp-40h]

  v9 = sub_16982C0();
  if ( v9 == *(void **)(a1 + 8) )
    return sub_16A13D0((__int64 *)(a1 + 8), (__int64 *)(a2 + 8), a3, a4, a5, a6);
  v10 = v9;
  if ( (unsigned __int8)sub_169DE70(a1) || (unsigned __int8)sub_169DE70(a2) )
  {
    v12 = (_BYTE *)(a1 + 8);
    if ( v10 == *(void **)(a1 + 8) )
    {
      sub_169CAA0((__int64)v12, 0, 0, 0, *(float *)&a4);
      return 1;
    }
    else
    {
      sub_16986F0(v12, 0, 0, 0);
      return 1;
    }
  }
  else if ( *(void **)(a1 + 8) == sub_1698270()
         && ((v13 = (char *)sub_16D40F0(qword_4FBB490)) == 0 ? (v14 = qword_4FBB490[2]) : (v14 = *v13), v14) )
  {
    v15 = (__int64 *)(a2 + 8);
    if ( v10 == *(void **)(a2 + 8) )
      sub_169C6E0(&v22, (__int64)v15);
    else
      sub_16986C0(&v22, v15);
    if ( v10 == v22 )
      sub_169C8D0((__int64)&v22, a4, a5, a6);
    else
      sub_1699490((__int64)&v22);
    result = sub_169FD40(a1, (__int64)v21, a3);
    if ( v10 == v22 )
    {
      v16 = v23;
      if ( v23 )
      {
        v17 = v23 + 32LL * *(_QWORD *)(v23 - 8);
        if ( v23 != v17 )
        {
          do
          {
            v17 -= 32;
            v19 = result;
            if ( v10 == *(void **)(v17 + 8) )
              sub_169DEB0((__int64 *)(v17 + 16));
            else
              sub_1698460(v17 + 8);
            LODWORD(result) = v19;
          }
          while ( v16 != v17 );
        }
        v20 = result;
        j_j_j___libc_free_0_0(v16 - 8);
        return v20;
      }
    }
    else
    {
      v18 = result;
      sub_1698460((__int64)&v22);
      return v18;
    }
  }
  else
  {
    return sub_169D430((__int16 **)(a1 + 8), (_BYTE *)(a2 + 8), a3);
  }
  return result;
}
