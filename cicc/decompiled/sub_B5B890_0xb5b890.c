// Function: sub_B5B890
// Address: 0xb5b890
//
__int64 __fastcall sub_B5B890(__int64 a1)
{
  __int64 ***v1; // r14
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r12
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rax
  unsigned int *v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rsi
  _QWORD *v13; // rdx
  __int64 v14; // rdx
  _QWORD *v15; // rax

  v1 = (__int64 ***)sub_B5B6B0(a1);
  if ( (unsigned int)*(unsigned __int8 *)v1 - 12 <= 1 )
    return sub_ACA8A0(v1[1]);
  if ( *((char *)v1 + 7) >= 0 )
    goto LABEL_15;
  v3 = sub_BD2BC0(v1);
  v5 = v3 + v4;
  if ( *((char *)v1 + 7) < 0 )
    v5 -= sub_BD2BC0(v1);
  v6 = v5 >> 4;
  if ( (_DWORD)v6 )
  {
    v7 = 0;
    v8 = 16LL * (unsigned int)v6;
    while ( 1 )
    {
      v9 = 0;
      if ( *((char *)v1 + 7) < 0 )
        v9 = sub_BD2BC0(v1);
      v10 = (unsigned int *)(v7 + v9);
      v11 = *((_DWORD *)v1 + 1) & 0x7FFFFFF;
      if ( *(_DWORD *)(*(_QWORD *)v10 + 8LL) == 5 )
        break;
      v7 += 16;
      if ( v7 == v8 )
        goto LABEL_16;
    }
    v12 = *(_QWORD *)(a1 + 32 * (2LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
    v13 = *(_QWORD **)(v12 + 24);
    if ( *(_DWORD *)(v12 + 32) > 0x40u )
      v13 = (_QWORD *)*v13;
    return (__int64)(&v1[4 * (unsigned int)v13])[4 * (v10[2] - v11)];
  }
  else
  {
LABEL_15:
    v11 = *((_DWORD *)v1 + 1) & 0x7FFFFFF;
LABEL_16:
    v14 = *(_QWORD *)(a1 + 32 * (2LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
    v15 = *(_QWORD **)(v14 + 24);
    if ( *(_DWORD *)(v14 + 32) > 0x40u )
      v15 = (_QWORD *)*v15;
    return (__int64)v1[4 * ((unsigned int)v15 - v11)];
  }
}
