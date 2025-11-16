// Function: sub_69C3A0
// Address: 0x69c3a0
//
__int64 __fastcall sub_69C3A0(__int64 a1, int *a2)
{
  __int64 result; // rax
  __int64 k; // rax
  __int64 m; // rbx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r15
  __int64 i; // rax
  __int64 j; // rdi
  unsigned __int64 v14; // rsi
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 **v20; // rbx
  __int64 v21; // rbx
  __int64 v22; // r13
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  unsigned int v27; // [rsp+8h] [rbp-E8h] BYREF
  unsigned int v28; // [rsp+Ch] [rbp-E4h] BYREF
  __int64 v29; // [rsp+10h] [rbp-E0h] BYREF
  __int64 *v30; // [rsp+18h] [rbp-D8h] BYREF
  _BYTE v31[208]; // [rsp+20h] [rbp-D0h] BYREF

  result = *(unsigned __int8 *)(a1 + 207);
  if ( (result & 0x10) == 0 )
  {
    v27 = 0;
    v30 = 0;
    v11 = qword_4D03C58;
    result = sub_8D24D0(a2);
    if ( !(_DWORD)result )
    {
      result = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
      if ( (*(_BYTE *)(result + 178) & 0x20) == 0 )
      {
        for ( i = *(_QWORD *)(a1 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        for ( j = *(_QWORD *)(i + 160); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        result = sub_68B310(j, &v27);
        if ( v27 != 32 )
        {
          sub_7296C0(&v28);
          sub_6E1DD0(&v29);
          v14 = (unsigned __int64)v31;
          v15 = 5;
          sub_6E1E00(5, v31, 0, 1);
          *(_DWORD *)(qword_4D03C50 + 18LL) |= 0x11080u;
          qword_4D03C58 = &v30;
          v20 = (__int64 **)**((_QWORD **)a2 + 21);
          if ( v20 )
          {
            while ( 1 )
            {
              if ( ((_BYTE)v20[12] & 1) != 0 )
              {
                v15 = (__int64)v20[5];
                v14 = v27;
                if ( (unsigned int)sub_69BFB0(v15, v27, v16, v17, v18, v19) )
                  break;
              }
              v20 = (__int64 **)*v20;
              if ( !v20 )
                goto LABEL_20;
            }
          }
          else
          {
LABEL_20:
            v21 = **(_QWORD **)(*(_QWORD *)a2 + 96LL);
            if ( !v21 )
            {
LABEL_34:
              sub_6E2B30(v15, v14);
              sub_6E1DF0(v29);
              sub_729730(v28);
              result = (__int64)v30;
              if ( v30 )
              {
                if ( (*(_BYTE *)(a1 + 193) & 0x12) == 2 )
                {
                  if ( (*(_BYTE *)(a1 + 195) & 3) != 1
                    && ((*(_BYTE *)(a1 + 206) & 8) == 0 || (*(_BYTE *)(a1 + 193) & 1) != 0) )
                  {
                    result = sub_6854C0(0xC1Cu, (FILE *)(a1 + 64), *v30);
                  }
                  *(_BYTE *)(a1 + 193) &= ~2u;
                }
              }
              else
              {
                result = *(unsigned __int8 *)(a1 + 193);
                if ( (result & 0x10) != 0 )
                {
                  result = (unsigned int)result | 2;
                  *(_BYTE *)(a1 + 193) = result;
                }
              }
              goto LABEL_31;
            }
            while ( 1 )
            {
              if ( *(_BYTE *)(v21 + 80) == 8 )
              {
                v22 = *(_QWORD *)(*(_QWORD *)(v21 + 88) + 120LL);
                if ( (unsigned int)sub_8D3410(v22) )
                  v22 = sub_8D40F0(v22);
                while ( *(_BYTE *)(v22 + 140) == 12 )
                  v22 = *(_QWORD *)(v22 + 160);
                v14 = v27;
                v15 = v22;
                if ( (unsigned int)sub_69BFB0(v22, v27, v23, v24, v25, v26) )
                  break;
              }
              v21 = *(_QWORD *)(v21 + 16);
              if ( !v21 )
                goto LABEL_34;
            }
          }
          sub_6E2B30(v15, v14);
          sub_6E1DF0(v29);
          result = sub_729730(v28);
        }
      }
    }
    *(_BYTE *)(a1 + 193) |= 0x22u;
    *(_BYTE *)(a1 + 206) |= 0x10u;
LABEL_31:
    qword_4D03C58 = v11;
    return result;
  }
  if ( (result & 0x20) == 0 )
  {
    for ( k = *(_QWORD *)(a1 + 152); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
      ;
    for ( m = *(_QWORD *)(k + 160); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
      ;
    result = sub_8D3EA0(m);
    if ( (_DWORD)result )
    {
      result = *(_QWORD *)(m + 168);
      if ( *(_DWORD *)(result + 24) == 1 )
        return (__int64)sub_6902E0(a1, a2, v7, v8, v9, v10);
    }
  }
  return result;
}
