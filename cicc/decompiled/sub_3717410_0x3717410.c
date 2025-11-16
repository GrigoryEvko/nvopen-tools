// Function: sub_3717410
// Address: 0x3717410
//
unsigned __int64 __fastcall sub_3717410(_DWORD *a1, int a2)
{
  __int64 (*v2)(void); // rax
  int v3; // eax
  unsigned int v5; // [rsp+1Bh] [rbp-15h]
  unsigned __int8 v6; // [rsp+1Fh] [rbp-11h]

  v2 = *(__int64 (**)(void))(*(_QWORD *)a1 + 56LL);
  if ( (char *)v2 == (char *)sub_3717150 )
    v3 = a1[36];
  else
    v3 = v2();
  if ( v3 == ((a2 + 1) & 0x7FFFFFFF) - 4096 )
  {
    v6 = 0;
  }
  else
  {
    v6 = 1;
    v5 = a2 + 1;
  }
  return ((unsigned __int64)v6 << 32) | v5;
}
