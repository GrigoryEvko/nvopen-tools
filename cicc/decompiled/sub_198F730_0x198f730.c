// Function: sub_198F730
// Address: 0x198f730
//
__int64 __fastcall sub_198F730(__int64 a1, __int64 *a2)
{
  __int64 *v3; // rax
  __int64 v4; // rdx
  __int64 *v5; // r14
  __int64 v6; // rsi
  __int64 *v7; // r13
  __int64 v9; // rax
  __int64 *v10; // rax
  int v11; // [rsp+4h] [rbp-3Ch] BYREF
  _QWORD v12[7]; // [rsp+8h] [rbp-38h] BYREF

  v12[0] = 0;
  v3 = *(__int64 **)(a1 + 16);
  if ( v3 == *(__int64 **)(a1 + 8) )
    v4 = *(unsigned int *)(a1 + 28);
  else
    v4 = *(unsigned int *)(a1 + 24);
  v5 = &v3[v4];
  if ( v3 != v5 )
  {
    while ( 1 )
    {
      v6 = *v3;
      v7 = v3;
      if ( (unsigned __int64)*v3 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v5 == ++v3 )
        goto LABEL_6;
    }
    if ( v5 != v3 )
    {
      do
      {
        v9 = sub_1368AA0(a2, v6);
        sub_16AF570(v12, v9);
        v10 = v7 + 1;
        if ( v7 + 1 == v5 )
          break;
        v6 = *v10;
        for ( ++v7; (unsigned __int64)*v10 >= 0xFFFFFFFFFFFFFFFELL; v7 = v10 )
        {
          if ( v5 == ++v10 )
            goto LABEL_6;
          v6 = *v10;
        }
      }
      while ( v5 != v7 );
    }
  }
LABEL_6:
  if ( (unsigned int)(*(_DWORD *)(a1 + 28) - *(_DWORD *)(a1 + 32)) > 1 )
  {
    sub_16AF710(&v11, dword_4FB10A0, 0x64u);
    sub_16AF520(v12, v11);
  }
  return v12[0];
}
