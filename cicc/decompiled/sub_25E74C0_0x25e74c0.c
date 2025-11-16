// Function: sub_25E74C0
// Address: 0x25e74c0
//
unsigned __int64 __fastcall sub_25E74C0(char *a1, char *a2, unsigned __int64 a3)
{
  unsigned __int64 result; // rax
  __int64 *v5; // rbx
  __int64 v6; // r14
  __int64 v7; // r15
  char *v8; // r13
  __int64 v9; // rcx
  unsigned __int64 v10; // rdx
  __int64 v11; // r8
  unsigned __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // [rsp+0h] [rbp-60h]
  __int64 v18; // [rsp+10h] [rbp-50h] BYREF
  __int64 v19; // [rsp+18h] [rbp-48h]
  unsigned __int64 v20; // [rsp+20h] [rbp-40h]

  result = a2 - a1;
  v5 = (__int64 *)a2;
  v17 = a2 - a1;
  if ( a2 - a1 > 24 )
  {
    v6 = 0xAAAAAAAAAAAAAAABLL * (v17 >> 3);
    v7 = (v6 - 2) / 2;
    v8 = &a1[8 * v7 + 8 * ((v6 - 2 + ((unsigned __int64)(v6 - 2) >> 63)) & 0xFFFFFFFFFFFFFFFELL)];
    while ( 1 )
    {
      v9 = *((_QWORD *)v8 + 1);
      v10 = *((_QWORD *)v8 + 2);
      v8 -= 24;
      v11 = *((_QWORD *)v8 + 3);
      v19 = v9;
      v20 = v10;
      v18 = v11;
      result = sub_25DC3F0((__int64)a1, v7, v6, &v18);
      if ( !v7 )
        break;
      --v7;
    }
  }
  if ( (unsigned __int64)a2 < a3 )
  {
    do
    {
      while ( 1 )
      {
        result = v5[2];
        v12 = *((_QWORD *)a1 + 2);
        if ( result < v12 )
          break;
        v5 += 3;
        if ( a3 <= (unsigned __int64)v5 )
          return result;
      }
      v13 = v5[1];
      v14 = *v5;
      v5[2] = v12;
      v5 += 3;
      v15 = *((_QWORD *)a1 + 1);
      v20 = result;
      v18 = v14;
      *(v5 - 2) = v15;
      v16 = *(_QWORD *)a1;
      v19 = v13;
      *(v5 - 3) = v16;
      result = sub_25DC3F0((__int64)a1, 0, 0xAAAAAAAAAAAAAAABLL * (v17 >> 3), &v18);
    }
    while ( a3 > (unsigned __int64)v5 );
  }
  return result;
}
