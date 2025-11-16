// Function: sub_1F12B80
// Address: 0x1f12b80
//
__int64 __fastcall sub_1F12B80(__int64 a1, unsigned int a2)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // r14
  __int64 v7; // r13
  int v8; // eax
  __int64 v9; // rsi
  char v10; // r13
  unsigned __int64 v11; // rax
  char v12; // dl
  __int64 result; // rax
  __int64 v14; // r8
  __int64 v15; // rbx
  __int64 v16; // r13
  __int64 i; // r14
  __int64 v18; // rdx
  int v19; // r12d
  unsigned __int8 *v20; // rdx
  unsigned int v21; // ecx
  unsigned int v22; // eax
  _BYTE *v23; // r9
  __int64 v24; // rdi
  _DWORD *v25; // rdx
  __int64 v26; // rax
  unsigned __int64 v27; // rax
  __int64 v28; // [rsp+8h] [rbp-58h]
  __int64 v29; // [rsp+10h] [rbp-50h]
  unsigned __int64 v30; // [rsp+20h] [rbp-40h] BYREF
  __int64 v31[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a1 + 264);
  v28 = 112LL * a2;
  v4 = v3 + v28;
  v5 = *(unsigned int *)(v3 + v28 + 32);
  v6 = *(_QWORD *)(v3 + v28 + 24);
  v30 = *(_QWORD *)(v3 + v28);
  v5 *= 16;
  v7 = v6 + v5;
  for ( v31[0] = *(_QWORD *)(v3 + v28 + 8); v7 != v6; v6 += 16 )
  {
    while ( 1 )
    {
      v8 = *(_DWORD *)(v3 + 112LL * *(unsigned int *)(v6 + 8) + 16);
      if ( v8 != -1 )
        break;
      v9 = *(_QWORD *)v6;
      v6 += 16;
      sub_16AF570(&v30, v9);
      if ( v7 == v6 )
        goto LABEL_8;
    }
    if ( v8 == 1 )
      sub_16AF570(v31, *(_QWORD *)v6);
  }
LABEL_8:
  v10 = *(_DWORD *)(v4 + 16) > 0;
  v11 = sub_16AF590(v31, *(_QWORD *)(a1 + 456));
  if ( v11 > v30 )
  {
    v27 = sub_16AF590((__int64 *)&v30, *(_QWORD *)(a1 + 456));
    if ( v27 > v31[0] )
    {
      *(_DWORD *)(v4 + 16) = 0;
      v12 = 0;
    }
    else
    {
      *(_DWORD *)(v4 + 16) = 1;
      v12 = 1;
    }
  }
  else
  {
    *(_DWORD *)(v4 + 16) = -1;
    v12 = 0;
  }
  result = 0;
  if ( v10 != v12 )
  {
    v14 = *(_QWORD *)(a1 + 264);
    v15 = v14 + v28;
    v16 = *(_QWORD *)(v14 + v28 + 24);
    for ( i = v16 + 16LL * *(unsigned int *)(v14 + v28 + 32); i != v16; v16 += 16 )
    {
      v18 = *(unsigned int *)(v16 + 8);
      v19 = *(_DWORD *)(v16 + 8);
      if ( *(_DWORD *)(v15 + 16) != *(_DWORD *)(v14 + 112 * v18 + 16) )
      {
        v20 = (unsigned __int8 *)(*(_QWORD *)(a1 + 512) + v18);
        v21 = *(_DWORD *)(a1 + 472);
        v22 = *v20;
        v23 = v20;
        if ( v22 >= v21 )
          goto LABEL_20;
        v24 = *(_QWORD *)(a1 + 464);
        while ( 1 )
        {
          v25 = (_DWORD *)(v24 + 4LL * v22);
          if ( v19 == *v25 )
            break;
          v22 += 256;
          if ( v21 <= v22 )
            goto LABEL_20;
        }
        if ( v25 == (_DWORD *)(v24 + 4LL * v21) )
        {
LABEL_20:
          *v23 = v21;
          v26 = *(unsigned int *)(a1 + 472);
          if ( (unsigned int)v26 >= *(_DWORD *)(a1 + 476) )
          {
            v29 = v14;
            sub_16CD150(a1 + 464, (const void *)(a1 + 480), 0, 4, v14, (int)v23);
            v26 = *(unsigned int *)(a1 + 472);
            v14 = v29;
          }
          *(_DWORD *)(*(_QWORD *)(a1 + 464) + 4 * v26) = v19;
          ++*(_DWORD *)(a1 + 472);
        }
      }
    }
    return 1;
  }
  return result;
}
