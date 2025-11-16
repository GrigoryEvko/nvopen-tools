// Function: sub_2B2A000
// Address: 0x2b2a000
//
__int64 __fastcall sub_2B2A000(__int64 a1, __int64 a2)
{
  int v2; // r11d
  __int64 v3; // rax
  _QWORD *v4; // rbx
  __int64 v5; // r12
  _QWORD *v6; // rdi
  _QWORD *v7; // rax
  __int64 v8; // r9
  _QWORD *v9; // r10
  unsigned int v10; // r11d
  __int64 v11; // r14
  _QWORD *v12; // r8
  __int64 v13; // r12
  int v14; // edx
  __int64 v15; // rax
  _DWORD *v16; // r15
  _DWORD *v17; // rax
  __int64 v19; // [rsp+8h] [rbp-48h] BYREF
  int v20[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v2 = *(_DWORD *)(a1 + 120);
  v3 = *(unsigned int *)(a1 + 8);
  v19 = a2;
  v4 = *(_QWORD **)a1;
  LODWORD(v5) = v2;
  if ( !v2 )
    LODWORD(v5) = v3;
  v6 = *(_QWORD **)a1;
  v20[0] = v5;
  v7 = sub_2B0CA10(v6, (__int64)&v4[v3], &v19);
  if ( v9 != v7 )
  {
    v11 = v19;
    v12 = v7;
    v13 = v10;
    while ( 1 )
    {
      if ( *v12 == v11 )
      {
        v14 = *(_DWORD *)(v8 + 152);
        v15 = v12 - v4;
        v20[0] = v15;
        if ( v14 )
          v20[0] = *(_DWORD *)(*(_QWORD *)(v8 + 144) + 4LL * (unsigned int)v15);
        if ( !v10 )
        {
LABEL_13:
          LODWORD(v5) = v20[0];
          return (unsigned int)v5;
        }
        v16 = *(_DWORD **)(v8 + 112);
        v17 = sub_2B0C950(v16, (__int64)&v16[v13], v20);
        if ( &v16[v13] != v17 )
          return (unsigned int)(v17 - v16);
      }
      if ( v9 == ++v12 )
        goto LABEL_13;
    }
  }
  return (unsigned int)v5;
}
