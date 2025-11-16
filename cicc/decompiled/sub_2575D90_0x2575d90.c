// Function: sub_2575D90
// Address: 0x2575d90
//
void __fastcall sub_2575D90(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // r13
  unsigned int v3; // r15d
  int v4; // esi
  __int64 v5; // r8
  int v6; // eax
  __int64 v7; // r14
  int v8; // eax
  unsigned int v9; // edx
  unsigned int v10; // r9d
  int v11; // eax
  int v12; // r15d
  unsigned int i; // ecx
  __int64 v14; // rsi
  int v15; // r10d
  unsigned int v16; // ecx
  bool v17; // al
  int v18; // eax
  int v19; // [rsp+10h] [rbp-60h]
  unsigned int v20; // [rsp+14h] [rbp-5Ch]
  unsigned int v21; // [rsp+18h] [rbp-58h]
  unsigned int v22; // [rsp+1Ch] [rbp-54h]
  __int64 v23; // [rsp+20h] [rbp-50h]
  __int64 v24; // [rsp+28h] [rbp-48h]
  __int64 v25[7]; // [rsp+38h] [rbp-38h] BYREF

  v1 = *(_QWORD *)(a1 + 32);
  v2 = v1 + 16LL * *(unsigned int *)(a1 + 40);
  if ( v2 == v1 )
    return;
  do
  {
    v3 = *(_DWORD *)(a1 + 24);
    if ( !v3 )
    {
      ++*(_QWORD *)a1;
      v25[0] = 0;
LABEL_4:
      v4 = 2 * v3;
      goto LABEL_5;
    }
    v7 = *(_QWORD *)(a1 + 8);
    v8 = sub_C4F140(v1);
    v9 = v3 - 1;
    v10 = *(_DWORD *)(v1 + 8);
    v5 = 0;
    v11 = (v3 - 1) & v8;
    v12 = 1;
    for ( i = v11; ; i = v9 & v16 )
    {
      v14 = v7 + 16LL * i;
      v15 = *(_DWORD *)(v14 + 8);
      if ( v10 == v15 )
      {
        if ( v10 <= 0x40 )
        {
          if ( *(_QWORD *)v1 == *(_QWORD *)v14 )
            goto LABEL_10;
        }
        else
        {
          v19 = *(_DWORD *)(v14 + 8);
          v20 = v10;
          v21 = i;
          v22 = v9;
          v23 = v5;
          v24 = v7 + 16LL * i;
          v17 = sub_C43C50(v1, (const void **)v14);
          v14 = v24;
          v5 = v23;
          v9 = v22;
          i = v21;
          v10 = v20;
          v15 = v19;
          if ( v17 )
            goto LABEL_10;
        }
      }
      if ( !v15 )
        break;
LABEL_15:
      v16 = v12 + i;
      ++v12;
    }
    if ( *(_QWORD *)v14 != -1 )
    {
      if ( *(_QWORD *)v14 == -2 && !v5 )
        v5 = v14;
      goto LABEL_15;
    }
    v3 = *(_DWORD *)(a1 + 24);
    if ( !v5 )
      v5 = v14;
    v18 = *(_DWORD *)(a1 + 16);
    ++*(_QWORD *)a1;
    v25[0] = v5;
    v6 = v18 + 1;
    if ( 4 * v6 >= 3 * v3 )
      goto LABEL_4;
    if ( v3 - (v6 + *(_DWORD *)(a1 + 20)) <= v3 >> 3 )
    {
      v4 = v3;
LABEL_5:
      sub_2575960(a1, v4);
      sub_2567ED0(a1, v1, v25);
      v5 = v25[0];
      v6 = *(_DWORD *)(a1 + 16) + 1;
    }
    *(_DWORD *)(a1 + 16) = v6;
    if ( (!*(_DWORD *)(v5 + 8) && *(_QWORD *)v5 == -1 || (--*(_DWORD *)(a1 + 20), *(_DWORD *)(v5 + 8) <= 0x40u))
      && *(_DWORD *)(v1 + 8) <= 0x40u )
    {
      *(_QWORD *)v5 = *(_QWORD *)v1;
      *(_DWORD *)(v5 + 8) = *(_DWORD *)(v1 + 8);
    }
    else
    {
      sub_C43990(v5, v1);
    }
LABEL_10:
    v1 += 16;
  }
  while ( v2 != v1 );
}
