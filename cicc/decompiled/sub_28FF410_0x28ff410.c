// Function: sub_28FF410
// Address: 0x28ff410
//
__int64 __fastcall sub_28FF410(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r8
  __int64 *v7; // rax
  int v9; // edx

  v6 = *(unsigned __int8 *)(a1 + 28);
  if ( !(_BYTE)v6 )
    goto LABEL_8;
  v7 = *(__int64 **)(a1 + 8);
  a4 = *(unsigned int *)(a1 + 20);
  a3 = &v7[a4];
  if ( v7 == a3 )
  {
LABEL_7:
    if ( (unsigned int)a4 >= *(_DWORD *)(a1 + 16) )
    {
LABEL_8:
      sub_C8CC70(a1, a2, (__int64)a3, a4, v6, a6);
      return v9 ^ 1u;
    }
    *(_DWORD *)(a1 + 20) = a4 + 1;
    *a3 = a2;
    ++*(_QWORD *)a1;
    return 0;
  }
  else
  {
    while ( a2 != *v7 )
    {
      if ( a3 == ++v7 )
        goto LABEL_7;
    }
    return *(unsigned __int8 *)(a1 + 28);
  }
}
