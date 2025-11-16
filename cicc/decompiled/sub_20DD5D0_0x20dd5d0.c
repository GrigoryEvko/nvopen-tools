// Function: sub_20DD5D0
// Address: 0x20dd5d0
//
__int64 __fastcall sub_20DD5D0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, _DWORD *a5)
{
  _QWORD *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r15
  _QWORD *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // r12
  unsigned int v12; // r13d
  __int64 v13; // rdx
  __int16 v14; // ax
  __int64 *v15; // rdx
  _QWORD *v16; // r12
  __int64 v17; // rax
  unsigned int v18; // r8d
  __int64 v20; // rax
  __int16 v21; // dx
  __int64 v22; // rax
  __int16 v23; // dx
  bool v24; // al
  __int64 v27; // [rsp+18h] [rbp-48h]
  unsigned int v28; // [rsp+28h] [rbp-38h]

  *a5 = 0;
  v5 = *(_QWORD **)(a1 + 112);
  v6 = (__int64)(*(_QWORD *)(a1 + 120) - (_QWORD)v5) >> 4;
  if ( (_DWORD)v6 )
  {
    v28 = -1;
    v7 = 0;
    v27 = (unsigned int)v6;
    while ( 1 )
    {
      v8 = &v5[2 * v7];
      v9 = *(_QWORD *)(*v8 + 8LL);
      if ( *a2 == v9 )
      {
        *a5 = v7;
        v5 = (_QWORD *)(16 * v7 + *(_QWORD *)(a1 + 112));
        goto LABEL_18;
      }
      v10 = v8[1];
      v11 = *(_QWORD *)(v9 + 32);
      v12 = 0;
      if ( v11 != v10 )
        break;
LABEL_42:
      v28 = v12;
      *a5 = v7;
LABEL_16:
      v5 = *(_QWORD **)(a1 + 112);
      if ( v27 == ++v7 )
      {
        v5 += 2 * (unsigned int)*a5;
        goto LABEL_18;
      }
    }
    while ( 1 )
    {
      v13 = *(_QWORD *)(v11 + 16);
      if ( *(_WORD *)v13 == 12 || *(_WORD *)v13 == 2 )
        goto LABEL_13;
      v14 = *(_WORD *)(v11 + 46);
      if ( (v14 & 4) != 0 || (v14 & 8) == 0 )
      {
        if ( (*(_QWORD *)(v13 + 8) & 0x10LL) == 0 )
        {
LABEL_28:
          v20 = *(_QWORD *)(v11 + 16);
          if ( *(_WORD *)v20 == 1 && (*(_BYTE *)(*(_QWORD *)(v11 + 32) + 64LL) & 8) != 0 )
            goto LABEL_33;
          v21 = *(_WORD *)(v11 + 46);
          if ( (v21 & 4) != 0 || (v21 & 8) == 0 )
          {
            if ( (*(_QWORD *)(v20 + 8) & 0x10000LL) != 0 )
              goto LABEL_33;
          }
          else if ( sub_1E15D00(v11, 0x10000u, 1) )
          {
LABEL_33:
            v12 += 2;
            goto LABEL_13;
          }
          v22 = *(_QWORD *)(v11 + 16);
          if ( *(_WORD *)v22 != 1 || (*(_BYTE *)(*(_QWORD *)(v11 + 32) + 64LL) & 0x10) == 0 )
          {
            v23 = *(_WORD *)(v11 + 46);
            if ( (v23 & 4) != 0 || (v23 & 8) == 0 )
              v24 = (*(_QWORD *)(v22 + 8) & 0x20000LL) != 0;
            else
              v24 = sub_1E15D00(v11, 0x20000u, 1);
            if ( !v24 )
            {
              ++v12;
              goto LABEL_13;
            }
          }
          goto LABEL_33;
        }
      }
      else if ( !sub_1E15D00(v11, 0x10u, 1) )
      {
        goto LABEL_28;
      }
      v12 += 10;
LABEL_13:
      if ( (*(_BYTE *)v11 & 4) != 0 )
      {
        v11 = *(_QWORD *)(v11 + 8);
        if ( v10 == v11 )
          goto LABEL_15;
      }
      else
      {
        while ( (*(_BYTE *)(v11 + 46) & 8) != 0 )
          v11 = *(_QWORD *)(v11 + 8);
        v11 = *(_QWORD *)(v11 + 8);
        if ( v10 == v11 )
        {
LABEL_15:
          if ( v12 > v28 )
            goto LABEL_16;
          goto LABEL_42;
        }
      }
    }
  }
LABEL_18:
  v15 = (__int64 *)v5[1];
  v16 = *(_QWORD **)(*v5 + 8LL);
  if ( !a3 || (unsigned int)((__int64)(v16[12] - v16[11]) >> 3) != 1 )
  {
    v17 = sub_20DCF10(a1, v16, v15, v16[5]);
    if ( v17 )
      goto LABEL_21;
    return 0;
  }
  v17 = sub_20DCF10(a1, v16, v15, *(_QWORD *)(a3 + 40));
  if ( !v17 )
    return 0;
LABEL_21:
  v18 = 1;
  *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 112) + 16LL * (unsigned int)*a5) + 8LL) = v17;
  *(_QWORD *)(*(_QWORD *)(a1 + 112) + 16LL * (unsigned int)*a5 + 8) = *(_QWORD *)(v17 + 32);
  if ( (_QWORD *)*a2 == v16 )
    *a2 = v17;
  return v18;
}
