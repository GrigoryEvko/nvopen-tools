// Function: sub_2C1B530
// Address: 0x2c1b530
//
unsigned __int64 __fastcall sub_2C1B530(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  unsigned __int8 *v7; // r12
  unsigned __int8 **v8; // rax

  v6 = a3;
  v7 = *(unsigned __int8 **)(a1 + 136);
  if ( !*(_BYTE *)(a3 + 108) )
  {
LABEL_8:
    sub_C8CC70(v6 + 80, (__int64)v7, a3, a4, a5, a6);
    return sub_2AD2480(v6, v7, a2);
  }
  v8 = *(unsigned __int8 ***)(a3 + 88);
  a4 = *(unsigned int *)(a3 + 100);
  a3 = (__int64)&v8[a4];
  if ( v8 == (unsigned __int8 **)a3 )
  {
LABEL_7:
    if ( (unsigned int)a4 < *(_DWORD *)(v6 + 96) )
    {
      *(_DWORD *)(v6 + 100) = a4 + 1;
      *(_QWORD *)a3 = v7;
      ++*(_QWORD *)(v6 + 80);
      return sub_2AD2480(v6, v7, a2);
    }
    goto LABEL_8;
  }
  while ( v7 != *v8 )
  {
    if ( (unsigned __int8 **)a3 == ++v8 )
      goto LABEL_7;
  }
  return sub_2AD2480(v6, v7, a2);
}
