// Function: sub_98A180
// Address: 0x98a180
//
__int64 __fastcall sub_98A180(unsigned __int8 *a1, __int64 a2)
{
  __int64 v4; // rdx
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v9; // rdx
  int v10; // eax
  char v11; // al
  __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // r14
  unsigned int i; // r15d
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned __int8 *v24; // r12
  __int64 v25; // r15
  _QWORD *v26; // r14
  _QWORD *j; // r12
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  int v31; // [rsp+Ch] [rbp-44h]
  __int64 v32[8]; // [rsp+10h] [rbp-40h] BYREF

  while ( 1 )
  {
    if ( (unsigned __int8)sub_BCAC40(*((_QWORD *)a1 + 1), 8) )
      return (__int64)a1;
    v5 = sub_BD5C60(a1, 8, v4);
    v6 = sub_BCB2B0(v5);
    v7 = sub_ACA8A0(v6);
    if ( (unsigned int)*a1 - 12 <= 1 )
      return v7;
    v32[0] = sub_9208B0(a2, *((_QWORD *)a1 + 1));
    v32[1] = v9;
    if ( !((unsigned __int64)(v32[0] + 7) >> 3) )
    {
      v29 = sub_BCB2B0(v5);
      return sub_ACADE0(v29);
    }
    if ( *a1 > 0x15u )
      return 0;
    if ( (unsigned __int8)sub_AC30F0(a1) )
      break;
    v10 = *a1;
    switch ( (_BYTE)v10 )
    {
      case 0x12:
        v11 = *(_BYTE *)(*((_QWORD *)a1 + 1) + 8LL);
        if ( v11 )
        {
          if ( v11 == 2 )
          {
            v13 = sub_BCB2D0(v5);
          }
          else
          {
            if ( v11 != 3 )
              return 0;
            v13 = sub_BCB2E0(v5);
          }
        }
        else
        {
          v13 = sub_BCB2C0(v5);
        }
        if ( !v13 )
          return 0;
        a1 = (unsigned __int8 *)sub_AD4C90(a1, v13, 0, v12, v14, v15);
        break;
      case 0x11:
        if ( (a1[32] & 7) != 0 )
          return 0;
        v24 = a1 + 24;
        if ( !(unsigned __int8)sub_C489C0(v24, 8) )
          return 0;
        sub_C44740(v32, v24);
        v7 = sub_ACCFD0(v5, v32);
        sub_969240(v32);
        return v7;
      case 5:
        if ( *((_WORD *)a1 + 1) != 48 )
          return 0;
        v16 = *((_QWORD *)a1 + 1);
        if ( *(_BYTE *)(v16 + 8) != 14 )
          return 0;
        v17 = sub_AE2980(a2, *(_DWORD *)(v16 + 8) >> 8);
        v18 = sub_BCD140(v5, *(unsigned int *)(v17 + 4));
        v19 = sub_96F3F0(*(_QWORD *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)], v18, 0, a2);
        if ( !v19 )
        {
          v10 = *a1;
LABEL_24:
          if ( (unsigned int)(v10 - 15) <= 1 )
          {
            v31 = sub_AC5290(a1);
            if ( v31 )
            {
              v20 = v7;
              for ( i = 0; i != v31; ++i )
              {
                v22 = sub_AD68C0(a1, i);
                v23 = sub_98A180(v22, a2);
                if ( v23 == v20 )
                {
                  if ( !v20 )
                    return 0;
                }
                else
                {
                  if ( !v20 || !v23 )
                    return 0;
                  if ( v7 == v20 )
                  {
                    v20 = v23;
                  }
                  else if ( v7 != v23 )
                  {
                    return 0;
                  }
                }
              }
              return v20;
            }
            return v7;
          }
          if ( (unsigned int)(v10 - 9) <= 2 )
          {
            v25 = v7;
            v26 = (_QWORD *)sub_986520((__int64)a1);
            for ( j = &v26[4 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)]; j != v26; v26 += 4 )
            {
              v28 = sub_98A180(*v26, a2);
              if ( v28 == v25 )
              {
                if ( !v25 )
                  return 0;
              }
              else
              {
                if ( !v25 || !v28 )
                  return 0;
                if ( v7 == v25 )
                {
                  v25 = v28;
                }
                else if ( v7 != v28 )
                {
                  return 0;
                }
              }
            }
            return v25;
          }
          return 0;
        }
        a1 = (unsigned __int8 *)v19;
        break;
      default:
        goto LABEL_24;
    }
  }
  v30 = sub_BCB2B0(v5);
  return sub_AD6530(v30);
}
