// Function: sub_16BE1E0
// Address: 0x16be1e0
//
void __fastcall sub_16BE1E0(__int64 a1, char *a2, __int64 a3)
{
  char *v3; // r8
  int *v4; // rdi
  __int64 v5; // r9
  __int64 v6; // r10
  __int64 v7; // r9
  __int64 v8; // r10

  v3 = *(char **)(a1 + 56);
  v4 = (int *)(a1 + 48);
  if ( v3 >= a2 && v3 <= &a2[a3] )
  {
    sub_16BDFB0(v4, v3, a3 - (v3 - a2));
    *(_QWORD *)(v7 + 56) = v8;
  }
  else
  {
    sub_16BDFB0(v4, a2, a3);
    *(_QWORD *)(v5 + 56) = v6;
  }
}
