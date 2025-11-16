// Function: sub_39A4F50
// Address: 0x39a4f50
//
__int64 __fastcall sub_39A4F50(__int64 *a1, __int64 a2, __int64 *a3, char a4)
{
  __int64 *v4; // rbx
  unsigned int v5; // eax
  __int64 v6; // r9
  __int64 v8; // rax
  unsigned int v9; // r12d
  __int64 **v10; // r15
  signed int v11; // r12d
  char *v12; // rax
  int v13; // r13d
  char v14; // cl
  __int64 v15; // rdx
  int v16; // edx
  int v17; // [rsp+8h] [rbp-38h]
  char v18; // [rsp+Fh] [rbp-31h]

  v4 = a3;
  v5 = *((_DWORD *)a3 + 2);
  if ( v5 > 0x40 )
  {
    v8 = sub_145CBF0(a1 + 11, 16, 16);
    v9 = *((_DWORD *)v4 + 2);
    *(_QWORD *)v8 = 0;
    v10 = (__int64 **)v8;
    *(_DWORD *)(v8 + 8) = 0;
    if ( v9 <= 0x40 )
    {
      v11 = v9 >> 3;
      v12 = (char *)sub_396DDB0(a1[24]);
      if ( !v11 )
        return sub_39A4C90(a1, a2, 28, v10);
    }
    else
    {
      v4 = (__int64 *)*v4;
      v11 = v9 >> 3;
      v12 = (char *)sub_396DDB0(a1[24]);
    }
    v13 = 0;
    v18 = *v12;
    v17 = v11 - 1;
    do
    {
      if ( v18 )
      {
        v14 = v17 - v13;
        v16 = v17 - v13 + 7;
        if ( v17 - v13 >= 0 )
          v16 = v17 - v13;
        v15 = v16 >> 3;
      }
      else
      {
        v14 = v13;
        v15 = v13 >> 3;
      }
      ++v13;
      sub_39A35E0((__int64)a1, (__int64 *)v10, 11, (unsigned __int8)((unsigned __int64)v4[v15] >> (8 * (v14 & 7u))));
    }
    while ( v13 < v11 );
    return sub_39A4C90(a1, a2, 28, v10);
  }
  v6 = *a3;
  if ( !a4 )
    v6 = v6 << (64 - (unsigned __int8)v5) >> (64 - (unsigned __int8)v5);
  return sub_39A37F0((__int64)a1, a2, a4, v6);
}
