// Function: sub_3573710
// Address: 0x3573710
//
__int64 __fastcall sub_3573710(__int64 a1, _QWORD *a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  unsigned __int8 v6; // dl
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rbx
  __int64 v11; // r12
  __int64 v12; // rsi
  unsigned __int8 *v13; // rsi
  __int64 v15; // r14
  int v16; // eax
  __int64 i; // r13
  unsigned __int64 *v18; // rcx
  unsigned __int64 v19; // rdx
  __int64 v20; // [rsp+8h] [rbp-78h]
  _QWORD *v22; // [rsp+18h] [rbp-68h]
  __int64 v23; // [rsp+20h] [rbp-60h]
  __int64 v24; // [rsp+28h] [rbp-58h]
  unsigned __int8 v25; // [rsp+38h] [rbp-48h]
  _QWORD v26[7]; // [rsp+48h] [rbp-38h] BYREF

  if ( *(_BYTE *)(a1 + 169) && !sub_BA8DC0((__int64)a2, (__int64)"llvm.debugify", 13) )
    return 0;
  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_33:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_50208C0 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_33;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_50208C0);
  v6 = 0;
  v20 = v5 + 176;
  v22 = (_QWORD *)a2[4];
  if ( v22 != a2 + 3 )
  {
    while ( 1 )
    {
      v25 = v6;
      v7 = (__int64)(v22 - 7);
      if ( !v22 )
        v7 = 0;
      v8 = sub_2EAA2D0(v20, v7);
      v6 = v25;
      if ( v8 )
      {
        v23 = v8 + 320;
        v24 = *(_QWORD *)(v8 + 328);
        if ( v8 + 320 != v24 )
          break;
      }
LABEL_22:
      v22 = (_QWORD *)v22[1];
      if ( a2 + 3 == v22 )
        return v6 | (unsigned int)sub_29C1CB0(a2);
    }
    while ( 1 )
    {
      v9 = *(_QWORD *)(v24 + 56);
      v10 = v24 + 40;
      if ( v24 + 48 != v9 )
        break;
LABEL_21:
      v24 = *(_QWORD *)(v24 + 8);
      if ( v23 == v24 )
        goto LABEL_22;
    }
    while ( 1 )
    {
      v11 = v9;
      v9 = *(_QWORD *)(v9 + 8);
      if ( (unsigned __int16)(*(_WORD *)(v11 + 68) - 14) <= 4u && (*(_DWORD *)(v11 + 40) & 0xFFFFFFu) > 1 )
        break;
      v12 = *(_QWORD *)(v11 + 56);
      if ( v12 )
      {
        v26[0] = 0;
        if ( (_QWORD *)(v11 + 56) != v26 )
        {
          sub_B91220(v11 + 56, v12);
          v13 = (unsigned __int8 *)v26[0];
          *(_QWORD *)(v11 + 56) = v26[0];
          if ( v13 )
            sub_B976B0((__int64)v26, v13, v11 + 56);
        }
LABEL_19:
        v6 = 1;
      }
      if ( v24 + 48 == v9 )
        goto LABEL_21;
    }
    v15 = v9;
    if ( (*(_BYTE *)v11 & 4) == 0 && (*(_BYTE *)(v11 + 44) & 8) != 0 )
    {
      do
      {
        v16 = *(_DWORD *)(v15 + 44);
        v15 = *(_QWORD *)(v15 + 8);
      }
      while ( (v16 & 8) != 0 );
    }
    if ( v15 != v11 )
    {
      for ( i = v9; ; i = *(_QWORD *)(i + 8) )
      {
        sub_2E31080(v10, v11);
        v18 = *(unsigned __int64 **)(v11 + 8);
        v19 = *(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL;
        *v18 = v19 | *v18 & 7;
        *(_QWORD *)(v19 + 8) = v18;
        *(_QWORD *)v11 &= 7uLL;
        *(_QWORD *)(v11 + 8) = 0;
        sub_2E310F0(v10);
        if ( v15 == i )
          break;
        v11 = i;
      }
    }
    goto LABEL_19;
  }
  return v6 | (unsigned int)sub_29C1CB0(a2);
}
