// Function: sub_2A89100
// Address: 0x2a89100
//
__int64 __fastcall sub_2A89100(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4)
{
  __int64 v4; // r12
  __int64 v5; // rax
  _QWORD *v6; // r15
  __int64 v10; // rax
  unsigned __int64 v11; // r15
  unsigned __int8 *v12; // rax
  unsigned __int8 *v13; // rax
  __int64 v14; // r12
  unsigned int v15; // r12d
  int v17; // edx
  __int64 v18; // rax
  _DWORD *v19; // rax
  int v20; // r12d
  __int64 v21; // rdi
  unsigned int v22; // edx
  unsigned __int8 *v23; // rax
  unsigned __int8 *v24; // rax
  unsigned int v25; // eax
  __int64 v26; // r10
  __int64 v27; // rbx
  unsigned __int8 *v28; // [rsp+0h] [rbp-50h]
  __int64 v29; // [rsp+8h] [rbp-48h]
  unsigned __int64 v30; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v31; // [rsp+18h] [rbp-38h]

  v29 = a2;
  v4 = *(_DWORD *)(a3 + 4) & 0x7FFFFFF;
  v5 = *(_QWORD *)(a3 + 32 * (2 - v4));
  if ( *(_BYTE *)v5 != 17 )
    return (unsigned int)-1;
  v6 = *(_QWORD **)(v5 + 24);
  if ( *(_DWORD *)(v5 + 32) > 0x40u )
    v6 = (_QWORD *)*v6;
  v10 = *(_QWORD *)(a3 - 32);
  if ( !v10 || *(_BYTE *)v10 || *(_QWORD *)(v10 + 24) != *(_QWORD *)(a3 + 80) )
    BUG();
  v11 = 8LL * (_QWORD)v6;
  if ( ((*(_DWORD *)(v10 + 36) - 243) & 0xFFFFFFFD) != 0 )
  {
    v12 = sub_BD3990(*(unsigned __int8 **)(a3 + 32 * (1 - v4)), a2);
    if ( *v12 <= 0x15u )
    {
      v28 = v12;
      v13 = sub_98ACB0(v12, 6u);
      v14 = (__int64)v13;
      if ( *v13 == 3
        && (v13[80] & 1) != 0
        && !sub_B2FC80((__int64)v13)
        && !(unsigned __int8)sub_B2F6B0(v14)
        && (*(_BYTE *)(v14 + 80) & 2) == 0 )
      {
        v24 = sub_BD3990(*(unsigned __int8 **)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)), 6);
        v15 = sub_2A88540(a1, a2, (__int64)v24, v11, (__int64)a4);
        if ( v15 != -1 )
        {
          v25 = sub_AE43F0((__int64)a4, *((_QWORD *)v28 + 1));
          v26 = (__int64)v28;
          v31 = v25;
          if ( v25 > 0x40 )
          {
            sub_C43690((__int64)&v30, (int)v15, 0);
            v26 = (__int64)v28;
          }
          else
          {
            v30 = (int)v15;
          }
          v27 = sub_971820(v26, a1, (__int64)&v30, a4);
          if ( v31 > 0x40 )
          {
            if ( v30 )
              j_j___libc_free_0_0(v30);
          }
          if ( v27 )
            return v15;
        }
      }
    }
    return (unsigned int)-1;
  }
  v17 = *(unsigned __int8 *)(a1 + 8);
  if ( (unsigned int)(v17 - 17) > 1 )
  {
    v18 = a1;
  }
  else
  {
    v18 = **(_QWORD **)(a1 + 16);
    LOBYTE(v17) = *(_BYTE *)(v18 + 8);
  }
  if ( (_BYTE)v17 == 14 )
  {
    a2 = *(_DWORD *)(v18 + 8) >> 8;
    v19 = sub_AE2980((__int64)a4, a2);
    v20 = *(_DWORD *)(a3 + 4);
    if ( !*((_BYTE *)v19 + 16) )
    {
      v4 = v20 & 0x7FFFFFF;
      goto LABEL_21;
    }
    v4 = v20 & 0x7FFFFFF;
    v21 = *(_QWORD *)(a3 + 32 * (1 - v4));
    if ( *(_BYTE *)v21 == 17 )
    {
      v22 = *(_DWORD *)(v21 + 32);
      if ( v22 <= 0x40 )
      {
        if ( !*(_QWORD *)(v21 + 24) )
          goto LABEL_21;
      }
      else if ( v22 == (unsigned int)sub_C444A0(v21 + 24) )
      {
        goto LABEL_21;
      }
    }
    return (unsigned int)-1;
  }
LABEL_21:
  v23 = sub_BD3990(*(unsigned __int8 **)(a3 - 32 * v4), a2);
  return sub_2A88540(a1, v29, (__int64)v23, v11, (__int64)a4);
}
