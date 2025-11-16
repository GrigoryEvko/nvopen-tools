// Function: sub_101C8A0
// Address: 0x101c8a0
//
unsigned __int8 *__fastcall sub_101C8A0(int a1, __int64 *a2, __int64 *a3, __m128i *a4, int a5)
{
  __int64 *v5; // r13
  unsigned int v8; // r12d
  unsigned __int8 *v9; // rbx
  unsigned __int8 *v10; // r12
  int v11; // eax
  __int64 *v12; // rdx
  unsigned __int8 *v13; // r12
  __int64 *v14; // r14
  __int64 *v16; // [rsp+8h] [rbp-48h]
  __int64 *v17; // [rsp+10h] [rbp-40h]
  __int64 *v18; // [rsp+18h] [rbp-38h]

  if ( !a5 )
    return 0;
  v5 = a2;
  v8 = a5 - 1;
  if ( *(_BYTE *)a2 == 86 || a3 == a2 )
  {
    v9 = sub_101AFF0(a1, (__int64 *)*(a2 - 8), a3, a4, v8);
    v17 = a2;
    v10 = sub_101AFF0(a1, (__int64 *)*(a2 - 4), a3, a4, v8);
  }
  else
  {
    v9 = sub_101AFF0(a1, a2, (__int64 *)*(a3 - 8), a4, v8);
    v17 = a3;
    v10 = sub_101AFF0(a1, a2, (__int64 *)*(a3 - 4), a4, v8);
  }
  if ( v9 != v10 && (!v9 || !(unsigned __int8)sub_1003090((__int64)a4, v9)) )
  {
    if ( v10 && (unsigned __int8)sub_1003090((__int64)a4, v10) )
      return v9;
    if ( v9 == (unsigned __int8 *)*(v17 - 8) && v10 == (unsigned __int8 *)*(v17 - 4) )
      return (unsigned __int8 *)v17;
    v16 = (__int64 *)*(v17 - 8);
    if ( (v10 != 0) != (v9 != 0) )
    {
      if ( v10 )
        v9 = v10;
      v11 = *v9;
      if ( (unsigned __int8)v11 > 0x1Cu && a1 == v11 - 29 && !(unsigned __int8)sub_B44920(v9) )
      {
        v12 = v16;
        if ( !v10 )
          v12 = (__int64 *)*(v17 - 4);
        if ( a2 == v17 )
        {
          v5 = v12;
          v12 = a3;
        }
        v13 = (v9[7] & 0x40) != 0
            ? (unsigned __int8 *)*((_QWORD *)v9 - 1)
            : &v9[-32 * (*((_DWORD *)v9 + 1) & 0x7FFFFFF)];
        v14 = *(__int64 **)v13;
        if ( *(__int64 **)v13 == v5 && v12 == *((__int64 **)v13 + 4) )
          return v9;
        v18 = v12;
        if ( sub_B46D50(v9) )
        {
          if ( *((__int64 **)v13 + 4) == v5 && v18 == v14 )
            return v9;
        }
      }
    }
    return 0;
  }
  return v10;
}
