// Function: sub_5E76F0
// Address: 0x5e76f0
//
__int64 __fastcall sub_5E76F0(_BYTE *a1, _DWORD *a2, _BYTE *a3, __int64 a4, int a5, unsigned int a6)
{
  _BYTE *v9; // rbx
  int i; // r10d
  __int64 v11; // r15
  bool v12; // r11
  char v13; // al
  __int64 result; // rax
  __int64 v15; // rdi
  char v16; // al
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // r13
  int v21; // esi
  __int64 v22; // rax
  unsigned int v23; // esi
  __int64 v24; // rax
  __int64 v25; // r13
  _BOOL4 v26; // eax
  __int64 v27; // rdx
  char v28; // al
  __int64 v29; // [rsp-10h] [rbp-70h]
  __int64 v30; // [rsp-10h] [rbp-70h]
  __int64 v31; // [rsp+8h] [rbp-58h]
  __int64 v32; // [rsp+8h] [rbp-58h]
  bool v33; // [rsp+10h] [rbp-50h]
  int v34; // [rsp+10h] [rbp-50h]
  bool v35; // [rsp+10h] [rbp-50h]
  int v36; // [rsp+10h] [rbp-50h]
  int v37; // [rsp+10h] [rbp-50h]
  int v38; // [rsp+10h] [rbp-50h]
  int v39; // [rsp+14h] [rbp-4Ch]
  bool v40; // [rsp+14h] [rbp-4Ch]
  int v41; // [rsp+14h] [rbp-4Ch]
  bool v42; // [rsp+14h] [rbp-4Ch]
  int v43; // [rsp+14h] [rbp-4Ch]
  bool v44; // [rsp+14h] [rbp-4Ch]
  bool v45; // [rsp+14h] [rbp-4Ch]
  int v46; // [rsp+14h] [rbp-4Ch]
  int v49; // [rsp+1Ch] [rbp-44h]
  int v50; // [rsp+1Ch] [rbp-44h]
  unsigned int v51; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v52; // [rsp+24h] [rbp-3Ch] BYREF
  _QWORD v53[7]; // [rsp+28h] [rbp-38h] BYREF

  v9 = a3;
  if ( (a3[140] & 0xFB) != 8 )
  {
    i = 0;
    v11 = *(_QWORD *)(*(_QWORD *)a3 + 96LL);
    goto LABEL_3;
  }
  for ( i = sub_8D4C10(a3, unk_4F077C4 != 2); v9[140] == 12; v9 = (_BYTE *)*((_QWORD *)v9 + 20) )
    ;
  v11 = *(_QWORD *)(*(_QWORD *)v9 + 96LL);
  if ( (i & 0xFFFFFFFE) == 0 )
    goto LABEL_3;
  v16 = *(_BYTE *)(v11 + 177);
  if ( (v16 & 0x20) != 0 )
  {
    a2[5] = 1;
    v16 = *(_BYTE *)(v11 + 177);
  }
  if ( (v16 & 0x40) == 0 )
    goto LABEL_3;
  if ( (a1[176] & 0x20) == 0 || !a4 || (v13 = *(_BYTE *)(a4 + 96), (v13 & 2) == 0) )
  {
    a2[3] = 1;
LABEL_3:
    v12 = a4 != 0;
    if ( dword_4F077B4 || !a4 )
    {
LABEL_8:
      if ( a2[5] )
        goto LABEL_12;
      if ( (v9[180] & 2) != 0 )
      {
        a2[5] = 1;
        if ( a2[9] )
        {
          v38 = i;
          v45 = v12;
          sub_686040(4, 1637, a1 + 64, a1, v9);
          v12 = v45;
          i = v38;
        }
        goto LABEL_12;
      }
      v21 = a2[1];
      v35 = v12;
      v41 = i;
      if ( a5 )
        v21 &= ~1u;
      v22 = sub_697B80((_DWORD)v9, v21, 0, i, (int)v9 + 64, (unsigned int)&v51, (__int64)&v52);
      i = v41;
      v12 = v35;
      if ( !v51 )
      {
        if ( v22 )
        {
          v37 = v41;
          v44 = v12;
          v32 = v22;
          v26 = sub_5E7660(v22);
          v12 = v44;
          i = v37;
          if ( !v26 )
          {
            v27 = v32;
            if ( !a6 || !dword_4D04434 || v52 || (*(_BYTE *)(*(_QWORD *)(v32 + 88) + 194LL) & 4) != 0 )
            {
              v28 = *(_BYTE *)(v32 + 80);
              if ( v28 == 16 )
              {
                v27 = **(_QWORD **)(v32 + 88);
                v28 = *(_BYTE *)(v27 + 80);
              }
              if ( v28 == 24 )
              {
                v27 = *(_QWORD *)(v27 + 88);
                v28 = *(_BYTE *)(v27 + 80);
              }
              if ( v28 == 10 && (*(_BYTE *)(*(_QWORD *)(v27 + 88) + 193LL) & 2) == 0 )
                a2[13] = 1;
LABEL_12:
              result = (unsigned int)a2[6];
              if ( (_DWORD)result )
                goto LABEL_13;
              result = dword_4D0446C;
              if ( !dword_4D0446C )
                goto LABEL_13;
              v33 = v12;
              v39 = i;
              v17 = sub_697B80((_DWORD)v9, 0, 1, i, (int)v9 + 64, (unsigned int)&v51, (__int64)&v52);
              i = v39;
              v18 = v17;
              result = v51;
              v12 = v33;
              if ( !v51 )
              {
                if ( v18 )
                {
                  v34 = v39;
                  v40 = v12;
                  v31 = v18;
                  result = sub_5E7660(v18);
                  v12 = v40;
                  i = v34;
                  if ( !(_DWORD)result )
                  {
                    v19 = v31;
                    if ( !a6
                      || !dword_4D04434
                      || v52
                      || (result = *(_QWORD *)(v31 + 88), (*(_BYTE *)(result + 194) & 4) != 0) )
                    {
                      result = *(unsigned __int8 *)(v31 + 80);
                      if ( (_BYTE)result == 16 )
                      {
                        v19 = **(_QWORD **)(v31 + 88);
                        result = *(unsigned __int8 *)(v19 + 80);
                      }
                      if ( (_BYTE)result == 24 )
                      {
                        v19 = *(_QWORD *)(v19 + 88);
                        result = *(unsigned __int8 *)(v19 + 80);
                      }
                      if ( (_BYTE)result == 10 )
                      {
                        result = *(_QWORD *)(v19 + 88);
                        if ( (*(_BYTE *)(result + 194) & 4) == 0 )
                          *(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 96LL) + 178LL) |= 4u;
                        if ( (*(_BYTE *)(result + 193) & 2) == 0 )
                          a2[14] = 1;
                      }
LABEL_13:
                      if ( (a1[176] & 0x20) == 0 || !v12 )
                        goto LABEL_16;
                      goto LABEL_15;
                    }
                  }
                }
                else if ( v52 )
                {
                  goto LABEL_13;
                }
              }
              a2[6] = 1;
              goto LABEL_13;
            }
          }
        }
        else if ( v52 )
        {
          goto LABEL_12;
        }
      }
      a2[5] = 1;
      if ( a2[9] )
      {
        v36 = i;
        v42 = v12;
        sub_686040(4, (unsigned int)(v51 == 0) + 1638, a1 + 64, a1, v9);
        v12 = v42;
        i = v36;
      }
      goto LABEL_12;
    }
    v13 = *(_BYTE *)(a4 + 96);
    goto LABEL_6;
  }
  if ( dword_4F077B4 )
  {
LABEL_7:
    v12 = a4 != 0;
    goto LABEL_8;
  }
LABEL_6:
  if ( (v13 & 3) != 2 )
    goto LABEL_7;
  v46 = i;
  result = sub_8E35E0(a4, a1);
  i = v46;
  if ( !(_DWORD)result )
    goto LABEL_7;
  if ( (a1[176] & 0x20) == 0 )
    goto LABEL_16;
LABEL_15:
  if ( (*(_BYTE *)(a4 + 96) & 2) != 0 )
    return result;
LABEL_16:
  if ( a2[3] )
    goto LABEL_20;
  if ( (v9[180] & 4) != 0 )
  {
    a2[3] = 1;
    if ( a2[8] )
    {
      v50 = i;
      result = sub_686040(4, 1640, a1 + 64, a1, v9);
      i = v50;
    }
    goto LABEL_20;
  }
  v23 = *a2 | i;
  v43 = i;
  if ( a5 )
    v23 = (*a2 | i) & 0xFFFFFFFE;
  v53[0] = 0;
  v24 = sub_697AE0((_DWORD)v9, v23, 0, (int)v9 + 64, (unsigned int)&v51, (unsigned int)v53, (__int64)&v52);
  i = v43;
  v25 = v24;
  result = v30;
  if ( !v51 && !v53[0] )
  {
    if ( v25 )
    {
      result = sub_5E7660(v25);
      i = v43;
      if ( !(_DWORD)result )
      {
        if ( !a6 || !dword_4D04434 || v52 || (result = *(_QWORD *)(v25 + 88), (*(_BYTE *)(result + 194) & 4) != 0) )
        {
          result = *(unsigned __int8 *)(v25 + 80);
          if ( (_BYTE)result == 16 )
          {
            v25 = **(_QWORD **)(v25 + 88);
            result = *(unsigned __int8 *)(v25 + 80);
          }
          if ( (_BYTE)result == 24 )
          {
            v25 = *(_QWORD *)(v25 + 88);
            result = *(unsigned __int8 *)(v25 + 80);
          }
          if ( (_BYTE)result == 10 )
          {
            result = *(_QWORD *)(v25 + 88);
            if ( (*(_BYTE *)(result + 193) & 2) == 0 )
              a2[11] = 1;
          }
          goto LABEL_20;
        }
      }
    }
    else if ( v52 )
    {
      goto LABEL_20;
    }
  }
  a2[3] = 1;
  if ( a2[8] )
  {
    v49 = i;
    result = sub_686040(4, (unsigned int)(v51 == 0) + 1641, a1 + 64, a1, v9);
    i = v49;
    if ( a2[4] )
      goto LABEL_21;
    goto LABEL_59;
  }
LABEL_20:
  if ( a2[4] )
    goto LABEL_21;
LABEL_59:
  result = (__int64)&dword_4D0446C;
  if ( dword_4D0446C )
  {
    v53[0] = 0;
    v20 = sub_697AE0((_DWORD)v9, i, 1, (int)v9 + 64, (unsigned int)&v51, (unsigned int)v53, (__int64)&v52);
    result = v29;
    if ( v51 || v53[0] )
    {
LABEL_62:
      a2[4] = 1;
      if ( a2[7] )
        goto LABEL_22;
      goto LABEL_26;
    }
    if ( v20 )
    {
      result = sub_5E7660(v20);
      if ( (_DWORD)result )
        goto LABEL_62;
      if ( a6 )
      {
        if ( dword_4D04434 )
        {
          if ( !v52 )
          {
            result = *(_QWORD *)(v20 + 88);
            if ( (*(_BYTE *)(result + 194) & 4) == 0 )
              goto LABEL_62;
          }
        }
      }
      result = *(unsigned __int8 *)(v20 + 80);
      if ( (_BYTE)result == 16 )
      {
        v20 = **(_QWORD **)(v20 + 88);
        result = *(unsigned __int8 *)(v20 + 80);
      }
      if ( (_BYTE)result == 24 )
      {
        v20 = *(_QWORD *)(v20 + 88);
        result = *(unsigned __int8 *)(v20 + 80);
      }
      if ( (_BYTE)result == 10 )
      {
        result = *(_QWORD *)(v20 + 88);
        if ( (*(_BYTE *)(result + 194) & 4) == 0 )
          *(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 96LL) + 178LL) |= 1u;
        if ( (*(_BYTE *)(result + 193) & 2) == 0 )
          a2[12] = 1;
      }
    }
    else
    {
      result = v52;
      if ( !v52 )
        goto LABEL_62;
      result = a6;
      if ( a6 )
      {
        result = (__int64)&dword_4D04434;
        if ( dword_4D04434 )
          result = v52;
      }
    }
  }
LABEL_21:
  if ( a2[7] )
  {
LABEL_22:
    if ( a2[2] && a2[3] && a2[4] )
      return result;
  }
LABEL_26:
  v15 = *(_QWORD *)(v11 + 24);
  if ( v15 )
  {
    if ( sub_5E7660(v15) )
    {
      if ( a2[10] )
      {
        if ( !a2[7] )
          sub_686040(5, 1623, a1 + 64, a1, v9);
      }
      a2[7] = 1;
      *((_QWORD *)a2 + 1) = 0x100000001LL;
      a2[4] = 1;
      return 0x100000001LL;
    }
    else
    {
      result = *(_QWORD *)(v11 + 24);
      if ( a6 && dword_4D04434 && result && (*(_BYTE *)(v11 + 177) & 2) == 0 )
      {
        a2[7] = 1;
      }
      else
      {
        result = *(_QWORD *)(result + 88);
        if ( (*(_BYTE *)(result + 193) & 2) == 0 )
          a2[15] = 1;
      }
    }
  }
  return result;
}
