// Function: sub_291D8F0
// Address: 0x291d8f0
//
char __fastcall sub_291D8F0(__int64 a1)
{
  __int64 v1; // rbp
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v6; // rdx
  unsigned __int8 *v7; // rax
  unsigned __int8 **v8; // rax
  unsigned __int8 *v9[6]; // [rsp-48h] [rbp-48h] BYREF
  unsigned __int8 *v10; // [rsp-18h] [rbp-18h] BYREF
  unsigned __int8 *v11; // [rsp-10h] [rbp-10h]
  __int64 v12; // [rsp-8h] [rbp-8h]

  v2 = *(_QWORD *)(a1 - 32);
  if ( !v2 || *(_BYTE *)v2 || *(_QWORD *)(v2 + 24) != *(_QWORD *)(a1 + 80) )
    BUG();
  if ( *(_DWORD *)(v2 + 36) == 68 )
    return sub_B59AF0(a1);
  v12 = v1;
  v3 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  v4 = *(_QWORD *)(*(_QWORD *)(a1 - 32 * v3) + 24LL);
  v9[0] = (unsigned __int8 *)v4;
  if ( *(_BYTE *)v4 == 4 )
  {
    if ( !*(_DWORD *)(v4 + 144) && !(unsigned __int8)sub_AF4500(*(_QWORD *)(*(_QWORD *)(a1 + 32 * (2 - v3)) + 24LL)) )
      return 1;
  }
  else if ( (unsigned __int8)(*(_BYTE *)v4 - 5) <= 0x1Fu )
  {
    return 1;
  }
  sub_B58DC0(&v10, v9);
  v6 = (__int64)v10;
  v9[2] = v10;
  v7 = v11;
  v9[1] = v11;
  v9[3] = v10;
  v9[4] = v10;
  v9[5] = v10;
  if ( v11 != v10 )
  {
    do
    {
      v8 = (unsigned __int8 **)(v6 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v6 & 4) != 0 )
      {
        if ( (unsigned int)**((unsigned __int8 **)*v8 + 17) - 12 <= 1 )
          goto LABEL_18;
        v6 = (unsigned __int64)(v8 + 1) | 4;
        v7 = (unsigned __int8 *)v6;
      }
      else
      {
        if ( (unsigned int)*v8[17] - 12 <= 1 )
        {
LABEL_18:
          v7 = (unsigned __int8 *)v6;
          return v11 != v7;
        }
        v7 = (unsigned __int8 *)(v8 + 18);
        v6 = (__int64)v7;
      }
    }
    while ( v11 != v7 );
  }
  return v11 != v7;
}
