// Function: sub_324A160
// Address: 0x324a160
//
__int64 __fastcall sub_324A160(__int64 *a1, __int64 a2, __int16 a3, __int64 *a4)
{
  __int64 v6; // rax
  unsigned __int64 **v7; // r15
  __int64 v8; // rdi
  char *v9; // rax
  char v10; // r13
  int v11; // r12d
  char v12; // cl
  __int64 v13; // rdx
  int v14; // edx
  int v17; // [rsp+18h] [rbp-38h]
  int v18; // [rsp+1Ch] [rbp-34h]

  v6 = sub_A777F0(0x10u, a1 + 11);
  v7 = (unsigned __int64 **)v6;
  if ( v6 )
  {
    *(_QWORD *)v6 = 0;
    *(_DWORD *)(v6 + 8) = 0;
  }
  v8 = a1[23];
  v18 = *((_DWORD *)a4 + 2) >> 3;
  if ( *((_DWORD *)a4 + 2) <= 0x40u )
  {
    v9 = (char *)sub_31DA930(v8);
    if ( !v18 )
      return sub_32498C0(a1, a2, a3, (__int64)v7);
  }
  else
  {
    a4 = (__int64 *)*a4;
    v9 = (char *)sub_31DA930(v8);
  }
  v10 = *v9;
  v11 = 0;
  v17 = v18 - 1;
  do
  {
    if ( v10 )
    {
      v12 = v17 - v11;
      v14 = v17 - v11 + 7;
      if ( v17 - v11 >= 0 )
        v14 = v17 - v11;
      v13 = v14 >> 3;
    }
    else
    {
      v12 = v11;
      v13 = v11 >> 3;
    }
    ++v11;
    sub_3249B00(a1, v7, 11, (unsigned __int8)((unsigned __int64)a4[v13] >> (8 * (v12 & 7u))));
  }
  while ( v11 < v18 );
  return sub_32498C0(a1, a2, a3, (__int64)v7);
}
