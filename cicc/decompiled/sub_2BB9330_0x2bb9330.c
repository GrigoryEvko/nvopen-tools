// Function: sub_2BB9330
// Address: 0x2bb9330
//
void __fastcall sub_2BB9330(__int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  __int64 v8; // rcx
  __int64 v9; // r15
  __int64 v10; // rbx

  v6 = (__int64)a2 - a1;
  if ( (__int64)a2 - a1 <= 896 )
  {
    sub_2BB7F80(a1, a2, a3, v6, a5, a6);
  }
  else
  {
    v8 = v6 >> 7;
    v9 = a1 + (v8 << 6);
    v10 = v8 << 6 >> 6;
    sub_2BB9330(a1, v9);
    sub_2BB9330(v9, a2);
    sub_2BB9190(a1, v9, (__int64)a2, v10, ((__int64)a2 - v9) >> 6, a3);
  }
}
