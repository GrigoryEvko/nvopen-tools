// Function: sub_3707BB0
// Address: 0x3707bb0
//
unsigned __int64 __fastcall sub_3707BB0(_DWORD *a1)
{
  __int64 (*v1)(void); // rax
  int v2; // eax
  unsigned int v4; // [rsp+Bh] [rbp-5h]
  unsigned __int8 v5; // [rsp+Fh] [rbp-1h]

  v1 = *(__int64 (**)(void))(*(_QWORD *)a1 + 56LL);
  if ( (char *)v1 == (char *)sub_3707AB0 )
    v2 = a1[20];
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
