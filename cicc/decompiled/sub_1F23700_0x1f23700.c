// Function: sub_1F23700
// Address: 0x1f23700
//
void __fastcall sub_1F23700(char *a1, char *a2, __int64 a3)
{
  __int64 v5; // rcx
  __int64 v6; // r10
  __int64 v7; // r8
  __int64 v8; // rbx
  __int64 *v9; // rax
  __int64 *v10; // r15
  __int64 *v11; // rsi
  __int64 *v12; // rax
  __int64 v13; // rdx
  __int64 *v14; // rax
  __int64 v15; // [rsp-50h] [rbp-50h]
  __int64 v16; // [rsp-48h] [rbp-48h]
  __int64 v17; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 && a2 != (char *)a3 )
  {
    v5 = 0xFFFFFFFFFFFFFFFLL;
    v6 = (a2 - a1) >> 3;
    v7 = (a3 - (__int64)a2) >> 3;
    if ( v6 + v7 <= 0xFFFFFFFFFFFFFFFLL )
      v5 = v6 + v7;
    if ( v6 + v7 <= 0 )
    {
LABEL_13:
      sub_1F21AE0(a1, a2, a3, v6, v7);
      j_j___libc_free_0(0, 0);
    }
    else
    {
      while ( 1 )
      {
        v8 = v5;
        v15 = v7;
        v16 = v6;
        v17 = v5;
        v9 = (__int64 *)sub_2207800(8 * v5, &unk_435FF63);
        v6 = v16;
        v7 = v15;
        v10 = v9;
        if ( v9 )
          break;
        v5 = v17 >> 1;
        if ( !(v17 >> 1) )
          goto LABEL_13;
      }
      v11 = &v9[v8];
      *v9 = *(_QWORD *)a1;
      v12 = v9 + 1;
      if ( v11 == v10 + 1 )
      {
        v14 = v10;
      }
      else
      {
        do
        {
          v13 = *(v12++ - 1);
          *(v12 - 1) = v13;
        }
        while ( v11 != v12 );
        v14 = &v10[v8 - 1];
      }
      *(_QWORD *)a1 = *v14;
      sub_1F226E0((__int64 *)a1, (__int64 *)a2, a3, v16, v15, v10, v17);
      j_j___libc_free_0(v10, v8 * 8);
    }
  }
}
