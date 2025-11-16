// Function: sub_833AE0
// Address: 0x833ae0
//
void __fastcall sub_833AE0(__int64 *a1)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 v4; // rdi
  __int64 v5; // r15
  _QWORD *v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // r8
  __int64 *v9; // r9
  __int64 v10; // rcx
  _QWORD *v11; // r14
  _QWORD *v12; // rsi
  __int64 v13; // [rsp+8h] [rbp-38h]

  v2 = a1[1];
  if ( v2 <= 1 )
  {
    v4 = 16;
    v3 = 2;
  }
  else
  {
    v3 = v2 + (v2 >> 1) + 1;
    v4 = 8 * v3;
  }
  v5 = *a1;
  v13 = a1[2];
  v6 = (_QWORD *)sub_823970(v4);
  v10 = v13;
  v11 = v6;
  if ( v13 > 0 )
  {
    v7 = (__int64 *)v5;
    v12 = &v6[v13];
    do
    {
      if ( v6 )
      {
        v10 = *v7;
        *v6 = *v7;
      }
      ++v6;
      ++v7;
    }
    while ( v12 != v6 );
  }
  sub_823A00(v5, 8 * v2, (__int64)v7, v10, v8, v9);
  *a1 = (__int64)v11;
  a1[1] = v3;
}
