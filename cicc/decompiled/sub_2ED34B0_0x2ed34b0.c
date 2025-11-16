// Function: sub_2ED34B0
// Address: 0x2ed34b0
//
void __fastcall sub_2ED34B0(__int64 *a1, __int64 *a2, __int64 a3, _QWORD *a4)
{
  __int64 v6; // rax
  __int64 *v7; // rbx
  __int64 v8; // r9
  __int64 v9; // [rsp+8h] [rbp-38h]

  if ( (char *)a2 - (char *)a1 <= 112 )
  {
    sub_2ED3280(a1, a2, a3, a4);
  }
  else
  {
    v6 = ((char *)a2 - (char *)a1) >> 4;
    v7 = &a1[v6];
    v9 = v6 * 8;
    sub_2ED34B0(a1, &a1[v6], a3, a4);
    sub_2ED34B0(v7, a2, a3, a4);
    sub_2ED2D90(a1, v7, (__int64)a2, v9 >> 3, a2 - v7, v8, a3, a4);
  }
}
