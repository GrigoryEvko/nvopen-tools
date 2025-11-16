// Function: sub_37173B0
// Address: 0x37173b0
//
unsigned __int64 __fastcall sub_37173B0(_DWORD *a1)
{
  __int64 (*v1)(void); // rax
  int v2; // eax
  unsigned int v4; // [rsp+Bh] [rbp-5h]
  unsigned __int8 v5; // [rsp+Fh] [rbp-1h]

  v1 = *(__int64 (**)(void))(*(_QWORD *)a1 + 56LL);
  if ( (char *)v1 == (char *)sub_3717150 )
    v2 = a1[36];
  else
    v2 = v1();
  if ( v2 )
  {
    v4 = 4096;
    v5 = 1;
  }
  else
  {
    v5 = 0;
  }
  return ((unsigned __int64)v5 << 32) | v4;
}
