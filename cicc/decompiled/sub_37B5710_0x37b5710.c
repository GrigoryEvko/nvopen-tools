// Function: sub_37B5710
// Address: 0x37b5710
//
unsigned __int64 __fastcall sub_37B5710(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  int v4; // edx
  unsigned int *v5; // rdx
  int v6; // ecx
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rax
  unsigned __int64 v10; // rsi
  unsigned __int64 result; // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rcx
  _BYTE *v18; // rsi
  _BYTE *v19; // rsi
  __int64 *v20; // [rsp+8h] [rbp-18h] BYREF

  v20 = a2;
  if ( !(unsigned __int8)sub_37B43A0((_QWORD *)a1, a2)
    || (v3 = *v20, (v4 = *(_DWORD *)(*v20 + 64)) != 0)
    && (v5 = (unsigned int *)(*(_QWORD *)(v3 + 40) + 40LL * (unsigned int)(v4 - 1)),
        *(_WORD *)(*(_QWORD *)(*(_QWORD *)v5 + 48LL) + 16LL * v5[2]) == 262) )
  {
    v14 = *(_QWORD *)(a1 + 160);
    v15 = *(_QWORD *)(v14 + 24);
    *(_QWORD *)(v14 + 40) = 1;
    if ( v15 )
      sub_37B5390(v15);
    v16 = *(_QWORD *)(a1 + 168);
    if ( *(_QWORD *)(a1 + 176) != v16 )
      *(_QWORD *)(a1 + 176) = v16;
    if ( !*v20 )
      goto LABEL_5;
    v6 = *(_DWORD *)(*v20 + 24);
    if ( v6 >= 0 )
      goto LABEL_5;
  }
  else
  {
    v6 = *(_DWORD *)(v3 + 24);
    if ( v6 >= 0 )
    {
LABEL_5:
      v7 = *(_QWORD *)(a1 + 160);
      v8 = *(_QWORD *)(v7 + 24);
      *(_QWORD *)(v7 + 40) = 1;
      if ( v8 )
        sub_37B5390(v8);
      v9 = *(_QWORD *)(a1 + 168);
      v10 = 0;
      if ( v9 != *(_QWORD *)(a1 + 176) )
        *(_QWORD *)(a1 + 176) = v9;
      goto LABEL_9;
    }
  }
  v17 = (unsigned int)~v6;
  if ( (unsigned int)v17 > 0x13 || ((1LL << v17) & 0x81700) == 0 )
    sub_37F1990(*(_QWORD *)(a1 + 160), *(_QWORD *)(*(_QWORD *)(a1 + 144) + 8LL) - 40 * v17);
  v18 = *(_BYTE **)(a1 + 176);
  if ( v18 == *(_BYTE **)(a1 + 184) )
  {
    sub_2ECAD30(a1 + 168, v18, &v20);
    v10 = (__int64)(*(_QWORD *)(a1 + 176) - *(_QWORD *)(a1 + 168)) >> 3;
  }
  else
  {
    if ( v18 )
    {
      *(_QWORD *)v18 = v20;
      v18 = *(_BYTE **)(a1 + 176);
    }
    v19 = v18 + 8;
    *(_QWORD *)(a1 + 176) = v19;
    v10 = (__int64)&v19[-*(_QWORD *)(a1 + 168)] >> 3;
  }
LABEL_9:
  result = **(unsigned int **)(a1 + 152);
  if ( result <= v10 )
  {
    v12 = *(_QWORD *)(a1 + 160);
    v13 = *(_QWORD *)(v12 + 24);
    *(_QWORD *)(v12 + 40) = 1;
    if ( v13 )
      sub_37B5390(v13);
    result = *(_QWORD *)(a1 + 168);
    if ( result != *(_QWORD *)(a1 + 176) )
      *(_QWORD *)(a1 + 176) = result;
  }
  return result;
}
