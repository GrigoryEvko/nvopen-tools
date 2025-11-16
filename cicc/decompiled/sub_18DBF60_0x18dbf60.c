// Function: sub_18DBF60
// Address: 0x18dbf60
//
__int64 __fastcall sub_18DBF60(__int64 a1, int a2, __int64 a3)
{
  unsigned int v3; // r14d
  __int64 *v6; // rax
  __int64 *v8; // rsi
  unsigned int v9; // edi
  __int64 *v10; // rcx

  v3 = 0;
  if ( a2 != 1 )
  {
    LOBYTE(v3) = *(_BYTE *)(a1 + 2) == 1;
    sub_18DB9D0(a1, 1);
    *(_BYTE *)(a1 + 8) = *(_BYTE *)a1;
    v6 = *(__int64 **)(a1 + 32);
    if ( *(__int64 **)(a1 + 40) != v6 )
    {
LABEL_3:
      sub_16CCBA0(a1 + 24, a3);
      goto LABEL_4;
    }
    v8 = &v6[*(unsigned int *)(a1 + 52)];
    v9 = *(_DWORD *)(a1 + 52);
    if ( v6 == v8 )
    {
LABEL_14:
      if ( v9 < *(_DWORD *)(a1 + 48) )
      {
        *(_DWORD *)(a1 + 52) = v9 + 1;
        *v8 = a3;
        ++*(_QWORD *)(a1 + 24);
        goto LABEL_4;
      }
      goto LABEL_3;
    }
    v10 = 0;
    while ( a3 != *v6 )
    {
      if ( *v6 == -2 )
        v10 = v6;
      if ( v8 == ++v6 )
      {
        if ( !v10 )
          goto LABEL_14;
        *v10 = a3;
        --*(_DWORD *)(a1 + 56);
        ++*(_QWORD *)(a1 + 24);
        break;
      }
    }
  }
LABEL_4:
  sub_18DB870((_BYTE *)a1);
  return v3;
}
