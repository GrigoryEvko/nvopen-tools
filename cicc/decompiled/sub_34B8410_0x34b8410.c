// Function: sub_34B8410
// Address: 0x34b8410
//
__int64 __fastcall sub_34B8410(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  unsigned int v3; // edx
  __int64 v4; // rax
  unsigned int *v6; // rsi
  unsigned int v7; // ecx
  __int64 v8; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r15
  int v13; // eax
  bool i; // al
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned int v17[13]; // [rsp+Ch] [rbp-34h] BYREF

  v3 = *(_DWORD *)(a2 + 8);
  if ( v3 )
  {
    v4 = *(unsigned int *)(a1 + 8);
    while ( 1 )
    {
      v6 = (unsigned int *)(*(_QWORD *)a2 + 4LL * v3 - 4);
      v7 = *v6 + 1;
      v8 = *(_QWORD *)(*(_QWORD *)a1 + 8 * v4 - 8);
      LOBYTE(v2) = *(_BYTE *)(v8 + 8) == 16 ? *(_QWORD *)(v8 + 32) > (unsigned __int64)v7 : v7 < *(_DWORD *)(v8 + 12);
      if ( (_BYTE)v2 )
        break;
      *(_DWORD *)(a2 + 8) = v3 - 1;
      v4 = (unsigned int)(*(_DWORD *)(a1 + 8) - 1);
      *(_DWORD *)(a1 + 8) = v4;
      v3 = *(_DWORD *)(a2 + 8);
      if ( !v3 )
        return 0;
    }
    *v6 = v7;
    v12 = sub_B501B0(
            *(_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 8),
            (unsigned int *)(*(_QWORD *)a2 + 4LL * *(unsigned int *)(a2 + 8) - 4),
            1);
    v13 = *(unsigned __int8 *)(v12 + 8);
    if ( v13 == 15 )
      goto LABEL_19;
LABEL_11:
    if ( v13 == 16 )
    {
      for ( i = *(_QWORD *)(v12 + 32) != 0; i; i = *(_DWORD *)(v12 + 12) != 0 )
      {
        v15 = *(unsigned int *)(a1 + 8);
        if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          sub_C8D5F0(a1, (const void *)(a1 + 16), v15 + 1, 8u, v10, v11);
          v15 = *(unsigned int *)(a1 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a1 + 8 * v15) = v12;
        ++*(_DWORD *)(a1 + 8);
        v16 = *(unsigned int *)(a2 + 8);
        if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          sub_C8D5F0(a2, (const void *)(a2 + 16), v16 + 1, 4u, v10, v11);
          v16 = *(unsigned int *)(a2 + 8);
        }
        *(_DWORD *)(*(_QWORD *)a2 + 4 * v16) = 0;
        ++*(_DWORD *)(a2 + 8);
        v17[0] = 0;
        v12 = sub_B501B0(v12, v17, 1);
        v13 = *(unsigned __int8 *)(v12 + 8);
        if ( v13 != 15 )
          goto LABEL_11;
LABEL_19:
        ;
      }
    }
  }
  else
  {
    return 0;
  }
  return v2;
}
