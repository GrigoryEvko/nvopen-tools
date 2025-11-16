// Function: sub_FDB420
// Address: 0xfdb420
//
__int64 __fastcall sub_FDB420(__int64 a1, __int64 a2, unsigned int *a3)
{
  __int64 v3; // rdx
  char *v4; // rsi

  v4 = (char *)sub_BD5D20(*(_QWORD *)(*(_QWORD *)(a2 + 136) + 8LL * *a3));
  *(_QWORD *)a1 = a1 + 16;
  if ( v4 )
  {
    sub_FDB1F0((__int64 *)a1, v4, (__int64)&v4[v3]);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_BYTE *)(a1 + 16) = 0;
  }
  return a1;
}
