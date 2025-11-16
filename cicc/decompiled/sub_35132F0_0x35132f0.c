// Function: sub_35132F0
// Address: 0x35132f0
//
void __fastcall sub_35132F0(__int64 *a1, __int64 *a2, __int64 a3, __int64 *a4)
{
  __int64 v6; // rax
  __int64 *v7; // rbx
  __int64 v8; // r9
  __int64 v9; // [rsp+8h] [rbp-38h]

  if ( (char *)a2 - (char *)a1 <= 112 )
  {
    sub_3511E50(a1, a2, a3, a4);
  }
  else
  {
    v6 = ((char *)a2 - (char *)a1) >> 4;
    v7 = &a1[v6];
    v9 = v6 * 8;
    sub_35132F0(a1, &a1[v6], a3, a4);
    sub_35132F0(v7, a2, a3, a4);
    sub_3513110(a1, v7, (__int64)a2, v9 >> 3, a2 - v7, v8, a3, a4);
  }
}
