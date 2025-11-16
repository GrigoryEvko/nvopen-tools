// Function: sub_271DB10
// Address: 0x271db10
//
__int64 __fastcall sub_271DB10(__int64 a1, int a2, __int64 a3)
{
  unsigned int v3; // r14d
  __int64 *v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  bool v9; // zf
  __int64 *v10; // rax

  v3 = 0;
  if ( a2 != 1 )
  {
    LOBYTE(v3) = *(_BYTE *)(a1 + 2) == 1;
    sub_271D520(a1, 1);
    v9 = *(_BYTE *)(a1 + 52) == 0;
    *(_BYTE *)(a1 + 8) = *(_BYTE *)a1;
    if ( v9 )
    {
LABEL_9:
      sub_C8CC70(a1 + 24, a3, (__int64)v5, v6, v7, v8);
      goto LABEL_7;
    }
    v10 = *(__int64 **)(a1 + 32);
    v6 = *(unsigned int *)(a1 + 44);
    v5 = &v10[v6];
    if ( v10 == v5 )
    {
LABEL_8:
      if ( (unsigned int)v6 < *(_DWORD *)(a1 + 40) )
      {
        *(_DWORD *)(a1 + 44) = v6 + 1;
        *v5 = a3;
        ++*(_QWORD *)(a1 + 24);
        goto LABEL_7;
      }
      goto LABEL_9;
    }
    while ( a3 != *v10 )
    {
      if ( v5 == ++v10 )
        goto LABEL_8;
    }
  }
LABEL_7:
  sub_271D2C0((_BYTE *)a1);
  return v3;
}
