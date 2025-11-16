// Function: sub_2AB4370
// Address: 0x2ab4370
//
__int64 __fastcall sub_2AB4370(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // rax
  __int64 v3; // rcx
  unsigned int v4; // edx
  __int64 v5; // r13
  __int64 v6; // rbx
  __int64 v7; // r12
  unsigned int v8; // eax
  unsigned int v9; // r12d
  __int64 v11; // rdx
  __int64 v12; // rax
  bool v13; // zf
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 *i; // rax
  __int64 v17; // r14
  __int64 v18; // rsi
  unsigned int v19; // eax
  unsigned int v20; // eax
  __int64 *v21; // [rsp+10h] [rbp-60h] BYREF
  __int64 *v22; // [rsp+18h] [rbp-58h]
  __int64 v23; // [rsp+20h] [rbp-50h]
  __int64 v24; // [rsp+28h] [rbp-48h]
  _QWORD v25[8]; // [rsp+30h] [rbp-40h] BYREF

  v1 = sub_B2BEC0(*(_QWORD *)(a1 + 488));
  v2 = *(unsigned int *)(a1 + 852);
  if ( *(_DWORD *)(a1 + 856) == (_DWORD)v2 && (v3 = *(_QWORD *)(a1 + 440), (v4 = *(_DWORD *)(v3 + 120)) != 0) )
  {
    v5 = *(_QWORD *)(v3 + 112);
    v6 = 0xFFFFFFFFLL;
    v7 = v5 + 184LL * v4;
    do
    {
      v8 = sub_BCB060(*(_QWORD *)(v5 + 64));
      if ( *(_DWORD *)(v5 + 176) <= v8 )
        v8 = *(_DWORD *)(v5 + 176);
      if ( (unsigned int)v6 > v8 )
        v6 = v8;
      v5 += 184;
    }
    while ( v7 != v5 );
    v9 = -1;
  }
  else
  {
    v11 = *(_QWORD *)(a1 + 840);
    if ( !*(_BYTE *)(a1 + 860) )
      v2 = *(unsigned int *)(a1 + 848);
    v21 = *(__int64 **)(a1 + 840);
    v22 = (__int64 *)(v11 + 8 * v2);
    sub_254BBF0((__int64)&v21);
    v12 = *(_QWORD *)(a1 + 832);
    v13 = *(_BYTE *)(a1 + 860) == 0;
    v23 = a1 + 832;
    v24 = v12;
    if ( v13 )
      v14 = *(unsigned int *)(a1 + 848);
    else
      v14 = *(unsigned int *)(a1 + 852);
    v25[0] = *(_QWORD *)(a1 + 840) + 8 * v14;
    v25[1] = v25[0];
    sub_254BBF0((__int64)v25);
    v15 = *(_QWORD *)(a1 + 832);
    v25[2] = a1 + 832;
    v6 = 8;
    v9 = -1;
    v25[3] = v15;
    for ( i = v21; (__int64 *)v25[0] != v21; i = v21 )
    {
      while ( 1 )
      {
        v17 = *i;
        v18 = *i;
        if ( (unsigned int)*(unsigned __int8 *)(*i + 8) - 17 <= 1 )
          v18 = **(_QWORD **)(v17 + 16);
        v19 = sub_9208B0(v1, v18);
        if ( v9 > v19 )
          v9 = v19;
        if ( (unsigned int)*(unsigned __int8 *)(v17 + 8) - 17 <= 1 )
          v17 = **(_QWORD **)(v17 + 16);
        v20 = sub_9208B0(v1, v17);
        if ( (unsigned int)v6 < v20 )
          v6 = v20;
        i = v21 + 1;
        v21 = i;
        if ( i != v22 )
          break;
LABEL_27:
        if ( (__int64 *)v25[0] == i )
          return (v6 << 32) | v9;
      }
      while ( (unsigned __int64)(*i + 2) <= 1 )
      {
        v21 = ++i;
        if ( v22 == i )
          goto LABEL_27;
      }
    }
  }
  return (v6 << 32) | v9;
}
