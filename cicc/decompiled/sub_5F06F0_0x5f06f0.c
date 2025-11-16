// Function: sub_5F06F0
// Address: 0x5f06f0
//
__int64 __fastcall sub_5F06F0(_BYTE *a1, __int64 a2, _BYTE *a3, __int64 a4, __int64 a5)
{
  _BYTE *v5; // r9
  __int64 v8; // r12
  __int64 v9; // rbx
  char v10; // al
  __int64 result; // rax
  __int64 v12; // rdi
  char v13; // al
  __int64 v14; // rbx
  _QWORD *v15; // rcx
  __int64 v16; // rdx
  _QWORD *v17; // r12
  __int64 v18; // rax
  __int64 k; // r13
  int v20; // eax
  char m; // al
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 n; // rdx
  _QWORD *v25; // rax
  __int64 v26; // rsi
  __int64 i; // rax
  __int64 v28; // r13
  __int64 v29; // rax
  __int64 v30; // r13
  int v31; // eax
  __int64 v32; // rax
  __int64 v33; // rax
  _QWORD *v34; // rdi
  _QWORD *v35; // rsi
  int v36; // eax
  int v37; // eax
  _QWORD *v38; // rbx
  __int64 v39; // rcx
  int v40; // eax
  _QWORD *v41; // rax
  __int64 v42; // rax
  int v43; // eax
  __int64 v44; // rax
  __int64 v45; // r8
  _QWORD *v46; // rcx
  __int64 v47; // rsi
  int v48; // eax
  _BOOL4 v49; // eax
  _BYTE *v50; // [rsp+0h] [rbp-70h]
  _BYTE *v51; // [rsp+8h] [rbp-68h]
  _BYTE *v52; // [rsp+8h] [rbp-68h]
  _BYTE *v53; // [rsp+8h] [rbp-68h]
  __int64 v54; // [rsp+8h] [rbp-68h]
  _BYTE *v55; // [rsp+8h] [rbp-68h]
  _BYTE *v56; // [rsp+8h] [rbp-68h]
  unsigned int v57; // [rsp+14h] [rbp-5Ch]
  _BYTE *v58; // [rsp+18h] [rbp-58h]
  int v59; // [rsp+18h] [rbp-58h]
  _BYTE *v60; // [rsp+18h] [rbp-58h]
  _BYTE *v61; // [rsp+18h] [rbp-58h]
  __int64 v62; // [rsp+20h] [rbp-50h]
  _QWORD *v63; // [rsp+20h] [rbp-50h]
  _QWORD *v64; // [rsp+20h] [rbp-50h]
  __int64 j; // [rsp+28h] [rbp-48h]
  __int64 v66; // [rsp+28h] [rbp-48h]
  char v67; // [rsp+28h] [rbp-48h]
  _QWORD v68[7]; // [rsp+38h] [rbp-38h] BYREF

  v5 = a3;
  v8 = *(_QWORD *)a1;
  v9 = *(_QWORD *)(*(_QWORD *)a1 + 88LL);
  if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)a1 + 80LL) - 10) > 1u )
  {
    v10 = *(_BYTE *)(a2 + 64) & 0x10 | *(_BYTE *)(v9 + 248) & 0xEF;
    *(_BYTE *)(v9 + 248) = v10;
    a3 = (_BYTE *)(*(_BYTE *)(a2 + 64) & 8);
    *(_BYTE *)(v9 + 248) = (unsigned __int8)a3 | v10 & 0xF7;
    v9 = *(_QWORD *)(v9 + 176);
  }
  result = *(unsigned __int8 *)(a2 + 64);
  if ( (result & 0x10) == 0 )
  {
    if ( (result & 8) == 0 )
      return result;
    if ( *(_BYTE *)(v8 + 80) == 20 )
    {
      v12 = 1807;
      goto LABEL_6;
    }
    result = *(unsigned __int8 *)(v9 + 174);
    if ( (_BYTE)result != 5 )
    {
      if ( (a1[8] & 8) != 0 )
      {
        v12 = 1779;
        goto LABEL_6;
      }
      if ( (_BYTE)result != 1 )
      {
        v12 = 1774;
        if ( (_BYTE)result == 2 )
        {
          *(_BYTE *)(v9 + 206) |= 8u;
          return result;
        }
        goto LABEL_6;
      }
      v26 = *(_QWORD *)(v8 + 64);
      LODWORD(v68[0]) = 0;
      for ( i = *(_QWORD *)(*(_QWORD *)(v8 + 88) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      result = *(_QWORD *)(i + 168);
      if ( (*(_BYTE *)(result + 16) & 1) != 0 )
        goto LABEL_54;
      v28 = *(_QWORD *)result;
      if ( !*(_QWORD *)result )
      {
        *(_BYTE *)(v9 + 206) |= 8u;
        if ( (a1[121] & 0x40) != 0 )
          goto LABEL_20;
        return result;
      }
      if ( *(_QWORD *)v28 || !sub_5E6640(*(_QWORD *)(v28 + 8), v26, v68) )
      {
LABEL_54:
        v12 = 1808;
LABEL_55:
        v5 = a1 + 48;
        goto LABEL_6;
      }
      if ( (*(_BYTE *)(v28 + 32) & 4) != 0 )
      {
        v12 = 1811;
        goto LABEL_55;
      }
LABEL_128:
      result = *(unsigned __int8 *)(v9 + 206);
      *(_BYTE *)(v9 + 206) |= 8u;
      if ( LODWORD(v68[0]) )
      {
        result = (unsigned int)result | 0x18;
        *(_BYTE *)(v9 + 206) = result;
      }
      return result;
    }
    if ( unk_4D041C0 )
    {
      v13 = *(_BYTE *)(v9 + 176);
      if ( (unsigned __int8)(v13 - 30) <= 4u || (unsigned __int8)(v13 - 16) <= 1u )
      {
        v14 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 88LL) + 152LL);
        for ( j = *(_QWORD *)(*(_QWORD *)a1 + 88LL); *(_BYTE *)(v14 + 140) == 12; v14 = *(_QWORD *)(v14 + 160) )
          ;
        v15 = qword_4F04C68;
        v16 = *(_QWORD *)(v14 + 168);
        v17 = *(_QWORD **)v16;
        v18 = qword_4F04C68[0] + 776LL * dword_4F04C64;
        if ( *(_BYTE *)(v18 + 4) == 6 )
        {
          v57 = 1;
          v63 = *(_QWORD **)(v18 + 208);
          goto LABEL_57;
        }
        for ( k = v17[1]; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
          ;
        v58 = v5;
        v62 = *(_QWORD *)(v14 + 168);
        v20 = sub_8D2FB0(k);
        v16 = v62;
        v5 = v58;
        if ( v20 )
        {
          v42 = sub_8D46C0(k);
          v16 = v62;
          v5 = v58;
          k = v42;
          for ( m = *(_BYTE *)(v42 + 140); m == 12; m = *(_BYTE *)(k + 140) )
            k = *(_QWORD *)(k + 160);
        }
        else
        {
          m = *(_BYTE *)(k + 140);
        }
        if ( (unsigned __int8)(m - 9) > 2u )
          goto LABEL_39;
        if ( (*(_BYTE *)(*(_QWORD *)a1 + 81LL) & 0x10) != 0 )
        {
          v15 = *(_QWORD **)(*(_QWORD *)a1 + 64LL);
          v63 = v15;
          if ( (_QWORD *)k != v15 )
          {
            if ( !v15 )
              goto LABEL_39;
            a5 = dword_4F07588;
            if ( !dword_4F07588 )
              goto LABEL_39;
            v22 = *(_QWORD *)(k + 32);
            if ( v15[4] != v22 || !v22 )
              goto LABEL_39;
            v63 = (_QWORD *)k;
            v57 = 0;
LABEL_57:
            if ( (*(_BYTE *)(v16 + 21) & 1) != 0 )
            {
              if ( (*(_BYTE *)(v16 + 18) & 0x7F) == 1 )
              {
                v59 = 0;
                if ( (*(_BYTE *)(v16 + 19) & 0xC0) == 0x80 )
                {
                  v55 = v5;
                  sub_6851C0(3099, v5);
                  v59 = 1;
                  v5 = v55;
                }
              }
              else
              {
                v51 = v5;
                sub_6851C0(2970, v5);
                v59 = 1;
                v5 = v51;
              }
              v29 = **(_QWORD **)(v14 + 168);
              if ( v29 && (*(_BYTE *)(v29 + 35) & 1) != 0 )
                goto LABEL_62;
              if ( !v17 )
              {
LABEL_63:
                v30 = *(_QWORD *)(v14 + 160);
                if ( *(_BYTE *)(j + 176) == 34 )
                {
                  while ( *(_BYTE *)(v30 + 140) == 12 )
                    v30 = *(_QWORD *)(v30 + 160);
                  if ( (*(_BYTE *)(j + 207) & 0x10) != 0 )
                  {
                    v56 = v5;
                    v43 = sub_8D3EA0(v30);
                    v5 = v56;
                    if ( !v43 || *(_DWORD *)(*(_QWORD *)(v30 + 168) + 24LL) != 1 )
                    {
                      sub_6851C0(2983, v56);
                      goto LABEL_40;
                    }
                  }
                  if ( ((v59 ^ 1) & v57) != 0 )
                  {
                    v32 = qword_4F04C68[0] + 776LL * dword_4F04C64;
                    *(_BYTE *)(*(_QWORD *)(v32 + 600) + 11LL) |= 2u;
                    if ( v59 )
                      goto LABEL_40;
                    goto LABEL_68;
                  }
                }
                else
                {
                  v53 = v5;
                  v31 = sub_8D29A0(*(_QWORD *)(v14 + 160));
                  v5 = v53;
                  if ( !v31 )
                  {
                    sub_6851C0(2969, v53);
                    goto LABEL_40;
                  }
                }
                if ( v59 )
                  goto LABEL_40;
                if ( !v57 )
                {
                  v40 = *(unsigned __int8 *)(j + 176);
                  if ( (_BYTE)v40 == 34 )
                  {
                    sub_69C3A0(j, v63, v16, v57, a5, v5);
                  }
                  else if ( (_BYTE)v40 == 30 )
                  {
                    sub_691E30(v63, j, v16, v57, a5, v5);
                  }
                  else if ( (unsigned __int8)(v40 - 31) <= 2u || (unsigned __int8)(v40 - 16) <= 1u )
                  {
                    sub_692200(v63, j, (unsigned int)(v40 - 31), v57, a5, v5);
                  }
                  goto LABEL_69;
                }
                v32 = qword_4F04C68[0] + 776LL * dword_4F04C64;
LABEL_68:
                *(_BYTE *)(*(_QWORD *)(v32 + 600) + 11LL) |= 1u;
LABEL_69:
                *(_WORD *)(j + 192) |= 0x280u;
                *(_BYTE *)(j + 206) |= 8u;
                return j;
              }
LABEL_84:
              v54 = v14;
              v38 = v17;
              v50 = v5;
              do
              {
                if ( !(unsigned int)sub_8D31A0(v38[1], v68)
                  || (_QWORD *)v68[0] != v63
                  && !(unsigned int)sub_8D97D0(v63, v68[0], 0, v39, a5)
                  && !(unsigned int)sub_8DD3B0(v68[0]) )
                {
                  v14 = v54;
                  sub_685360(2968, v50);
                  v59 = 1;
                  v5 = v50;
                  goto LABEL_63;
                }
                v38 = (_QWORD *)*v38;
              }
              while ( v38 );
              v14 = v54;
              v5 = v50;
              goto LABEL_63;
            }
            v33 = **(_QWORD **)(v14 + 168);
            if ( v33 && (*(_BYTE *)(v33 + 35) & 1) != 0 )
            {
LABEL_62:
              v52 = v5;
              sub_6851C0(3213, v5);
              v59 = 1;
              v5 = v52;
              goto LABEL_63;
            }
            v34 = (_QWORD *)v17[1];
            v35 = *(_QWORD **)(*v17 + 8LL);
            if ( v34 != v35 )
            {
              v60 = v5;
              v36 = sub_8D97D0(v34, v35, 0, v15, a5);
              v5 = v60;
              if ( !v36 )
              {
LABEL_83:
                v59 = 0;
                goto LABEL_84;
              }
              v35 = (_QWORD *)v17[1];
            }
            if ( v35 == v63 || (v61 = v5, v37 = sub_8D97D0(v63, v35, 0, v15, a5), v5 = v61, v37) )
            {
              v59 = 0;
              goto LABEL_63;
            }
            goto LABEL_83;
          }
        }
        else
        {
          v41 = *(_QWORD **)(j + 232);
          if ( !v41 )
          {
LABEL_39:
            sub_6851C0(2967, v5);
LABEL_40:
            *(_BYTE *)(a2 + 64) &= 0xF9u;
            *(_BYTE *)(*(_QWORD *)a1 + 81LL) &= ~2u;
            *(_WORD *)(j + 192) &= 0xDF7Fu;
            *(_BYTE *)(j + 202) &= ~0x80u;
            *(_BYTE *)(j + 172) = 1;
            return j;
          }
          while ( 1 )
          {
            v15 = (_QWORD *)v41[1];
            if ( (_QWORD *)k == v15 )
              break;
            v41 = (_QWORD *)*v41;
            if ( !v41 )
              goto LABEL_39;
          }
          v63 = (_QWORD *)v41[1];
        }
        v57 = 0;
        goto LABEL_57;
      }
    }
    v12 = 1779;
    if ( (a1[8] & 8) != 0 )
      goto LABEL_6;
    v12 = 1774;
    if ( *(_BYTE *)(v9 + 176) != 15 )
      goto LABEL_6;
    v23 = *(_QWORD *)(v8 + 64);
    LODWORD(v68[0]) = 0;
    for ( n = *(_QWORD *)(*(_QWORD *)(v8 + 88) + 152LL); *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
      ;
    v25 = *(_QWORD **)(n + 168);
    if ( *v25 )
    {
      if ( !*(_QWORD *)*v25 )
      {
        v64 = (_QWORD *)*v25;
        v66 = n;
        v44 = sub_72D600(v23);
        v46 = v64;
        v47 = *(_QWORD *)(v66 + 160);
        v67 = *(_BYTE *)(*(_QWORD *)(v66 + 168) + 18LL);
        if ( v44 == v47 || (v48 = sub_8D97D0(v44, v47, 0, v64, v45), v46 = v64, v48) )
        {
          if ( (v67 & 0x7F) != 0 )
          {
            if ( dword_4F077C4 != 2 || unk_4F07778 <= 202001 )
              goto LABEL_48;
            v49 = sub_5E6640(v46[1], v23, v68);
            LODWORD(v68[0]) = 1;
          }
          else
          {
            v49 = sub_5E6640(v46[1], v23, v68);
          }
          if ( v49 )
            goto LABEL_128;
        }
      }
    }
LABEL_48:
    v5 = a1 + 48;
    v12 = 1806;
    goto LABEL_6;
  }
  result = (unsigned __int8)a1[127];
  if ( (result & 0x10) == 0 )
  {
    v12 = 1775;
LABEL_6:
    result = sub_6851C0(v12, v5);
    *(_BYTE *)(a2 + 64) &= 0xF9u;
    *(_BYTE *)(v8 + 81) &= ~2u;
    *(_WORD *)(v9 + 192) &= 0xDF7Fu;
    *(_BYTE *)(v9 + 202) &= ~0x80u;
    *(_BYTE *)(v9 + 172) = 1;
    return result;
  }
  if ( (result & 0x20) != 0 )
  {
    v12 = 1812;
    goto LABEL_6;
  }
  *(_WORD *)(v9 + 192) |= 0x2080u;
  *(_BYTE *)(v9 + 206) |= 0x10u;
  if ( *(_BYTE *)(v9 + 174) == 1 )
  {
    result = sub_72F310(v9, 1, a3, a4, a5, v5);
    if ( (_DWORD)result )
    {
      result = sub_732860(v9);
      if ( !(_DWORD)result )
LABEL_20:
        *(_BYTE *)(v9 + 194) |= 2u;
    }
  }
  return result;
}
