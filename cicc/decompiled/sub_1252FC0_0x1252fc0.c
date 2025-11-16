// Function: sub_1252FC0
// Address: 0x1252fc0
//
__int64 __fastcall sub_1252FC0(_QWORD *a1)
{
  __int64 *v1; // r13
  __int64 v2; // rdx
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rdi
  void (*v7)(void); // rax

  v1 = a1 + 14;
  v2 = a1[18];
  *a1 = off_49E6768;
  a1[14] = off_49E66F0;
  v4 = a1[16];
  v5 = a1[21] + v4 - v2;
  if ( v5 )
  {
    sub_CB6C70((__int64)(a1 + 14), v5);
    v2 = a1[18];
    v4 = a1[16];
  }
  if ( v2 != v4 )
    sub_CB5AE0(v1);
  sub_CB5840((__int64)v1);
  v6 = a1[13];
  if ( v6 )
  {
    v7 = *(void (**)(void))(*(_QWORD *)v6 + 8LL);
    if ( (char *)v7 == (char *)sub_106DB80 )
    {
      v5 = 8;
      j_j___libc_free_0(v6, 8);
    }
    else
    {
      v7();
    }
  }
  sub_E8EC10((__int64)a1, v5);
  return j_j___libc_free_0(a1, 264);
}
