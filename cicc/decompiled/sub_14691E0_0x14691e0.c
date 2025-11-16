// Function: sub_14691E0
// Address: 0x14691e0
//
__int64 __fastcall sub_14691E0(__int64 a1, __int64 a2)
{
  unsigned int v3; // esi
  __int64 v4; // rcx
  unsigned int v5; // edi
  unsigned int v6; // edx
  __int64 *v7; // rax
  __int64 v8; // r8
  __int64 v10; // r13
  char v11; // r12
  __int64 v12; // r15
  __int64 v13; // rbx
  __int64 v14; // r13
  int v15; // r10d
  __int64 *v16; // r9
  unsigned int v17; // edx
  __int64 v18; // r8
  int v19; // edx
  int v20; // eax
  int v21; // r9d
  __int64 v23; // [rsp+10h] [rbp-60h]
  __int64 v24; // [rsp+18h] [rbp-58h]
  __int64 *v25; // [rsp+28h] [rbp-48h] BYREF
  __int64 v26; // [rsp+30h] [rbp-40h] BYREF
  __int16 v27; // [rsp+38h] [rbp-38h]

  v23 = a2;
  v3 = *(_DWORD *)(a1 + 712);
  v4 = *(_QWORD *)(a1 + 696);
  if ( !v3 )
  {
    v10 = *(_QWORD *)(v23 + 32);
    v24 = *(_QWORD *)(v23 + 40);
    if ( v10 == v24 )
    {
      v27 = 257;
      v14 = a1 + 688;
      v26 = v23;
      goto LABEL_35;
    }
    goto LABEL_6;
  }
  v5 = v3 - 1;
  v6 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v4 + 16LL * v6);
  v8 = *v7;
  if ( a2 == *v7 )
  {
LABEL_3:
    if ( v7 != (__int64 *)(v4 + 16LL * v3) )
      return *((unsigned __int16 *)v7 + 4);
    v10 = *(_QWORD *)(v23 + 32);
    v24 = *(_QWORD *)(v23 + 40);
    if ( v24 == v10 )
      goto LABEL_33;
LABEL_6:
    v11 = 1;
    while ( 1 )
    {
      v12 = *(_QWORD *)(*(_QWORD *)v10 + 48LL);
      v13 = *(_QWORD *)v10 + 40LL;
      if ( v13 != v12 )
        break;
LABEL_18:
      v10 += 8;
      if ( v24 == v10 )
      {
        LOBYTE(v27) = 1;
        HIBYTE(v27) = v11;
        v4 = *(_QWORD *)(a1 + 696);
        v3 = *(_DWORD *)(a1 + 712);
        v14 = a1 + 688;
        v26 = v23;
        if ( v3 )
        {
          v5 = v3 - 1;
          goto LABEL_21;
        }
LABEL_35:
        v3 = 0;
        ++*(_QWORD *)(a1 + 688);
LABEL_36:
        v3 *= 2;
        goto LABEL_37;
      }
    }
    while ( 1 )
    {
      if ( !v12 )
        BUG();
      if ( *(_BYTE *)(v12 - 8) == 55 )
        break;
      if ( (unsigned __int8)sub_15F3040(v12 - 24) || (unsigned __int8)sub_15F3330(v12 - 24) )
      {
LABEL_11:
        v11 = 0;
LABEL_12:
        v12 = *(_QWORD *)(v12 + 8);
        if ( v13 == v12 )
          goto LABEL_18;
      }
      else
      {
        v12 = *(_QWORD *)(v12 + 8);
        if ( v13 == v12 )
          goto LABEL_18;
      }
    }
    if ( !(unsigned __int8)sub_15F32D0(v12 - 24) && (*(_BYTE *)(v12 - 6) & 1) == 0 )
      goto LABEL_12;
    goto LABEL_11;
  }
  v20 = 1;
  while ( v8 != -8 )
  {
    v21 = v20 + 1;
    v6 = v5 & (v20 + v6);
    v7 = (__int64 *)(v4 + 16LL * v6);
    v8 = *v7;
    if ( v23 == *v7 )
      goto LABEL_3;
    v20 = v21;
  }
  v10 = *(_QWORD *)(v23 + 32);
  v24 = *(_QWORD *)(v23 + 40);
  if ( v10 != v24 )
    goto LABEL_6;
LABEL_33:
  v14 = a1 + 688;
  v26 = v23;
  v27 = 257;
LABEL_21:
  v15 = 1;
  v16 = 0;
  v17 = v5 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
  v7 = (__int64 *)(v4 + 16LL * v17);
  v18 = *v7;
  if ( v23 == *v7 )
    return *((unsigned __int16 *)v7 + 4);
  while ( v18 != -8 )
  {
    if ( !v16 && v18 == -16 )
      v16 = v7;
    v17 = v5 & (v15 + v17);
    v7 = (__int64 *)(v4 + 16LL * v17);
    v18 = *v7;
    if ( v23 == *v7 )
      return *((unsigned __int16 *)v7 + 4);
    ++v15;
  }
  if ( v16 )
    v7 = v16;
  ++*(_QWORD *)(a1 + 688);
  v19 = *(_DWORD *)(a1 + 704) + 1;
  if ( 4 * v19 >= 3 * v3 )
    goto LABEL_36;
  if ( v3 - (v19 + *(_DWORD *)(a1 + 708)) <= v3 >> 3 )
  {
LABEL_37:
    sub_1469020(v14, v3);
    sub_145FBC0(v14, &v26, &v25);
    v7 = v25;
    v23 = v26;
    v19 = *(_DWORD *)(a1 + 704) + 1;
  }
  *(_DWORD *)(a1 + 704) = v19;
  if ( *v7 != -8 )
    --*(_DWORD *)(a1 + 708);
  *v7 = v23;
  *((_WORD *)v7 + 4) = v27;
  return *((unsigned __int16 *)v7 + 4);
}
