// Function: sub_642710
// Address: 0x642710
//
__int64 __fastcall sub_642710(__int64 *a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // r8
  __int64 v5; // r15
  int v6; // ebx
  __int64 v7; // r12
  __int64 result; // rax
  char v9; // dl
  __int64 v10; // r14
  char v11; // dl
  __int64 v12; // rdx
  __int64 v13; // rsi
  __int64 v14; // r9
  int v15; // edi
  int v16; // ebx
  __int64 v17; // rdi
  char v18; // al
  char v19; // al
  int v20; // edx
  int v21; // eax
  __int64 v22; // rax
  __int64 i; // rbx
  __int64 v24; // rdx
  __int64 v25; // r12
  __int64 v26; // rdi
  __int64 v27; // r8
  __int64 v28; // r12
  __int64 v29; // rdx
  __int64 v30; // rdi
  __int64 v31; // rsi
  __int64 v32; // rdi
  __int64 v33; // rax
  int v34; // r9d
  __int64 v35; // [rsp-8h] [rbp-98h]
  __int64 v36; // [rsp+0h] [rbp-90h]
  __int64 v37; // [rsp+8h] [rbp-88h]
  __int64 v38; // [rsp+10h] [rbp-80h]
  __int64 v39; // [rsp+10h] [rbp-80h]
  __int64 v40; // [rsp+18h] [rbp-78h]
  __int64 v41; // [rsp+18h] [rbp-78h]
  _BOOL4 v42; // [rsp+20h] [rbp-70h]
  __int16 v43; // [rsp+26h] [rbp-6Ah]
  int v44; // [rsp+28h] [rbp-68h]
  unsigned int v45; // [rsp+2Ch] [rbp-64h]
  __int64 v46; // [rsp+30h] [rbp-60h]
  int v47; // [rsp+38h] [rbp-58h]
  int v48; // [rsp+3Ch] [rbp-54h]
  unsigned __int8 v49; // [rsp+3Ch] [rbp-54h]
  int v50; // [rsp+40h] [rbp-50h] BYREF
  char v51[4]; // [rsp+44h] [rbp-4Ch] BYREF
  __int64 v52; // [rsp+48h] [rbp-48h] BYREF
  char v53[8]; // [rsp+50h] [rbp-40h] BYREF
  __int64 v54[7]; // [rsp+58h] [rbp-38h] BYREF

  v46 = a2;
  v3 = sub_8D2310(a1[7]);
  v5 = *a1;
  v6 = v3;
  v7 = *(_QWORD *)(*a1 + 24);
  if ( !v7 || ((*(_BYTE *)(v5 + 18) & 2) != 0 || !*(_QWORD *)(v5 + 32)) && (*(_DWORD *)(v5 + 16) & 0x10004) == 0 )
  {
    if ( (a1[8] & 0x30) != 0 )
    {
      a2 = 32;
      result = sub_7D5DD0(*a1, 32);
      v7 = *(_QWORD *)(v5 + 24);
    }
    else
    {
      if ( dword_4F04C34 )
      {
        a2 = *(_QWORD *)(*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C34 + 184) + 32LL);
        result = sub_7D4A40(*a1, a2, 32);
      }
      else
      {
        a2 = *a1;
        result = sub_7D4600(unk_4F07288, v5, 32);
      }
      v7 = *(_QWORD *)(v5 + 24);
    }
    if ( (*(_BYTE *)(v5 + 17) & 0x40) == 0 )
    {
      *(_BYTE *)(v5 + 16) &= ~0x80u;
      *(_QWORD *)(v5 + 24) = 0;
    }
    if ( !v7 )
      return result;
    v42 = 0;
    LOBYTE(result) = *(_BYTE *)(v7 + 82);
LABEL_16:
    v11 = *(_BYTE *)(v7 + 80);
    v10 = v7;
    if ( v11 != 16 )
    {
LABEL_17:
      if ( v11 == 24 )
        v7 = *(_QWORD *)(v7 + 88);
      goto LABEL_19;
    }
LABEL_7:
    v10 = v7;
    v11 = *(_BYTE *)(**(_QWORD **)(v7 + 88) + 80LL);
    v7 = **(_QWORD **)(v7 + 88);
    goto LABEL_17;
  }
  v42 = (a1[8] & 1) == 0;
  result = *(unsigned __int8 *)(v7 + 82);
  if ( (result & 8) != 0 )
    goto LABEL_16;
  v9 = *(_BYTE *)(v7 + 80);
  if ( v9 == 24 )
    return result;
  v10 = *(_QWORD *)(*a1 + 24);
  if ( v9 == 16 )
    goto LABEL_7;
LABEL_19:
  v12 = 0;
  if ( (result & 4) == 0 )
    LOBYTE(v12) = (unsigned int)sub_880920(v7, a2, 0) != 0;
  result = *(unsigned __int8 *)(v10 + 80);
  v47 = *(_DWORD *)(v10 + 40);
  v13 = qword_4F04C68[0];
  v45 = *(_DWORD *)(qword_4F04C68[0] + 776LL * *((int *)a1 + 10));
  if ( dword_4F077BC && (_BYTE)result == 24 )
  {
    LOBYTE(v14) = *(_BYTE *)(v7 + 80);
    if ( (_BYTE)v14 == 7 )
    {
      if ( dword_4F077C4 == 2 )
        goto LABEL_26;
      v12 = (unsigned int)qword_4F077B4;
      if ( !(_DWORD)qword_4F077B4 || (*((_BYTE *)a1 + 65) & 8) != 0 )
        goto LABEL_26;
      goto LABEL_144;
    }
    v14 = 24;
    goto LABEL_102;
  }
  v14 = *(unsigned __int8 *)(v10 + 80);
  v15 = ((_BYTE)result - 7) & 0xFB;
  if ( (((_BYTE)result - 7) & 0xFB) != 0 )
  {
    LOBYTE(v15) = (_BYTE)result == 17;
    LOBYTE(v4) = (_BYTE)result == 20;
    v12 = (unsigned int)v4 | v15 | (unsigned int)v12;
LABEL_102:
    if ( !(_BYTE)v12 )
    {
      if ( dword_4F077C4 == 2 && (_BYTE)v14 == 24 )
      {
        result = *(unsigned __int8 *)(v7 + 80);
        if ( (_BYTE)result == 20 || (_BYTE)result == 11 )
        {
          if ( v47 == v45 )
            a1[2] = v10;
        }
        else if ( (_BYTE)result == 7 )
        {
          result = (__int64)&dword_4F04C58;
          if ( dword_4F04C58 == -1 )
          {
            v29 = *(_QWORD *)(v7 + 88);
            result = *(_BYTE *)(v29 + 88) & 0x70;
            if ( (_BYTE)result == 48 )
            {
              result = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 9) & 0xE;
              if ( (_BYTE)result == 6 )
              {
                v30 = *(_QWORD *)(v29 + 120);
                v31 = a1[7];
                if ( v30 == v31 || (result = sub_8D97D0(v30, v31, 0, (unsigned int)dword_4F077C4, v4), (_DWORD)result) )
                  result = sub_881DB0(v10);
              }
            }
          }
        }
      }
      if ( (a1[8] & 0x30) != 0 )
      {
        result = *(unsigned int *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
        if ( *(_DWORD *)(v10 + 40) != (_DWORD)result )
          a1[3] = v10;
      }
      return result;
    }
  }
  if ( dword_4F077C4 != 2 )
  {
    v12 = (unsigned int)qword_4F077B4;
    if ( !(_DWORD)qword_4F077B4 )
      goto LABEL_26;
    if ( (_BYTE)result != 17 && (*((_BYTE *)a1 + 65) & 8) == 0 )
    {
LABEL_144:
      if ( (_BYTE)result == 11 )
      {
        v49 = v14;
        v13 = *(_QWORD *)(*(_QWORD *)(v10 + 88) + 104LL);
        result = sub_736C60(20, v13);
        v14 = v49;
        if ( result )
          goto LABEL_63;
      }
LABEL_26:
      v13 = v45;
      if ( v47 != v45 )
      {
        v12 = *((unsigned __int8 *)a1 + 64);
        if ( (v12 & 1) == 0 )
          goto LABEL_28;
      }
LABEL_52:
      v16 = 0;
LABEL_53:
      if ( *((_DWORD *)a1 + 10) == dword_4F04C34 )
      {
        a1[1] = v10;
        v17 = v10;
      }
      else
      {
        v21 = *(unsigned __int8 *)(v10 + 80);
        v12 = (unsigned int)(v21 - 10);
        if ( (unsigned __int8)(v21 - 10) <= 1u
          || (_BYTE)v21 == 17
          || (result = *(_QWORD *)(v10 + 88), (*(_BYTE *)(result + 89) & 1) == 0) )
        {
          a1[1] = v10;
          v17 = v10;
        }
        else
        {
          v17 = a1[1];
          if ( !v17 )
          {
            if ( (a1[8] & 0x30) == 0 )
              return result;
            result = qword_4F04C68[0] + 776LL * dword_4F04C64;
            if ( *(_DWORD *)result == *(_DWORD *)(v10 + 40) )
              return result;
LABEL_40:
            a1[3] = v10;
LABEL_41:
            if ( v17 && a1[2] && (!v16 || a1[4]) )
              a1[2] = 0;
            return result;
          }
        }
      }
LABEL_32:
      v18 = *(_BYTE *)(v17 + 80);
      if ( v18 == 16 )
      {
        v17 = **(_QWORD **)(v17 + 88);
        v18 = *(_BYTE *)(v17 + 80);
      }
      if ( v18 == 24 )
        v17 = *(_QWORD *)(v17 + 88);
      v19 = sub_880920(v17, v13, v12);
      v17 = a1[1];
      v20 = 4 * (v19 & 1);
      result = v20 | *((_BYTE *)a1 + 65) & 0xFBu;
      *((_BYTE *)a1 + 65) = v20 | *((_BYTE *)a1 + 65) & 0xFB;
      if ( !v10 )
        goto LABEL_41;
      LOBYTE(v12) = *((_BYTE *)a1 + 64);
LABEL_38:
      if ( (v12 & 0x30) == 0 )
        goto LABEL_41;
      result = *(unsigned int *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
      if ( *(_DWORD *)(v10 + 40) == (_DWORD)result )
        goto LABEL_41;
      goto LABEL_40;
    }
  }
LABEL_63:
  if ( !v6 || (_BYTE)v14 == 7 )
    goto LABEL_26;
  v12 = *((unsigned __int8 *)a1 + 64);
  if ( (v12 & 2) != 0 )
  {
    v22 = a1[9];
    v13 = *(_QWORD *)v22;
    v37 = *(_QWORD *)v22;
    v43 = *(_WORD *)(v22 + 42);
    v36 = *(_QWORD *)(*(_QWORD *)(v22 + 32) + 16LL);
  }
  else
  {
    v36 = 0;
    v43 = 0;
    v37 = 0;
  }
  if ( v47 != v45 )
  {
    result = *(unsigned __int8 *)(v10 + 80);
    if ( (_BYTE)result == 17 )
    {
      v10 = *(_QWORD *)(v10 + 88);
      if ( !v10 )
      {
LABEL_162:
        v12 &= 1u;
        if ( (_DWORD)v12 )
          goto LABEL_139;
        goto LABEL_159;
      }
      goto LABEL_70;
    }
LABEL_155:
    v48 = 0;
    goto LABEL_71;
  }
  a1[2] = v10;
  result = *(unsigned __int8 *)(v10 + 80);
  if ( (_BYTE)result != 17 )
    goto LABEL_155;
  a1[4] = v10;
  v10 = *(_QWORD *)(v10 + 88);
  if ( !v10 )
    goto LABEL_139;
LABEL_70:
  v48 = 1;
  LODWORD(result) = *(unsigned __int8 *)(v10 + 80);
LABEL_71:
  v44 = 0;
  for ( i = v10; ; LODWORD(result) = *(unsigned __int8 *)(i + 80) )
  {
    v24 = (unsigned int)result;
    v25 = i;
    if ( (_BYTE)result == 16 )
    {
      v25 = **(_QWORD **)(i + 88);
      v24 = *(unsigned __int8 *)(v25 + 80);
    }
    if ( (_BYTE)v24 == 24 )
      v25 = *(_QWORD *)(v25 + 88);
    if ( (*(_BYTE *)(i + 82) & 4) != 0 )
    {
      if ( (_BYTE)result != 24 )
        goto LABEL_78;
    }
    else if ( (unsigned int)sub_880920(v25, v13, v24) || *(_BYTE *)(i + 80) != 24 )
    {
      goto LABEL_78;
    }
    result = *(unsigned __int8 *)(v5 + 18);
    if ( (result & 1) == 0 )
    {
      v12 = 0;
      if ( (result & 2) == 0 )
        v12 = *(_QWORD *)(v5 + 32);
      if ( *(_QWORD *)(v25 + 64) != v12 )
        goto LABEL_86;
    }
LABEL_78:
    v12 = *(unsigned __int8 *)(v25 + 80);
    result = a1[8] & 2;
    if ( (_BYTE)v12 == 20 )
    {
      if ( (_BYTE)result )
      {
        v12 = *(_QWORD *)(v25 + 88);
        result = *(_QWORD *)(v12 + 328);
        if ( *(_WORD *)(result + 42) == v43 )
        {
          v39 = *(_QWORD *)(v12 + 176);
          v41 = *(_QWORD *)(*(_QWORD *)(result + 32) + 16LL);
          v13 = v37;
          result = sub_89B3C0(*(_QWORD *)result, v37, 0, 0, 0, 8);
          if ( (_DWORD)result )
          {
            v13 = v36;
            result = sub_739400(v41, v36);
            if ( (_DWORD)result )
            {
              v13 = a1[7];
              v32 = *(_QWORD *)(v39 + 152);
              if ( v32 == v13 || (result = sub_8DED30(v32, v13, 1314824), (_DWORD)result) )
              {
                v13 = *(_QWORD *)(v46 + 400);
                result = sub_739400(*(_QWORD *)(v39 + 216), v13);
                if ( (_DWORD)result )
                {
                  a1[1] = v25;
                  a1[2] = 0;
                  return result;
                }
              }
            }
          }
        }
      }
      else
      {
        v44 = 1;
      }
      goto LABEL_86;
    }
    if ( !(_BYTE)result && (*(_BYTE *)(v5 + 18) & 1) == 0 )
    {
      v12 = (unsigned int)(v12 - 10);
      if ( (unsigned __int8)v12 > 1u )
      {
        a1[1] = 0;
        a1[2] = 0;
        return result;
      }
      v13 = a1[7];
      v14 = *(_QWORD *)(v46 + 400);
      result = *(_QWORD *)(v25 + 88);
      v26 = *(_QWORD *)(result + 152);
      v27 = *(_QWORD *)(result + 216);
      if ( v26 == v13
        || (v38 = *(_QWORD *)(result + 216),
            v40 = *(_QWORD *)(v46 + 400),
            result = sub_8DED30(v26, v13, 1314824),
            v14 = v40,
            v27 = v38,
            (_DWORD)result) )
      {
        if ( v14 == v27 )
          break;
        v13 = v14;
        result = sub_739400(v27, v14);
        if ( (_DWORD)result )
          break;
      }
    }
LABEL_86:
    if ( v48 )
    {
      i = *(_QWORD *)(i + 8);
      if ( i )
        continue;
    }
    if ( v44 )
    {
      result = (__int64)&dword_4D04340;
      v12 = *(unsigned int *)(v5 + 16);
      if ( !dword_4D04340 && (v12 & 0x10001) == 0 )
      {
        v13 = v45;
        if ( v47 == v45 || (a1[8] & 1) != 0 )
          goto LABEL_139;
LABEL_159:
        if ( v42 )
        {
          a1[1] = 0;
          return result;
        }
LABEL_139:
        v17 = a1[1];
        v16 = 0;
        v10 = 0;
        if ( !v17 )
          return result;
        goto LABEL_32;
      }
      v52 = 0;
      while ( 1 )
      {
        v12 &= 0x10001u;
        result = *(unsigned __int8 *)(v10 + 80);
        v28 = v10;
        if ( (_DWORD)v12 )
        {
          if ( (_BYTE)result == 16 )
          {
            v28 = **(_QWORD **)(v10 + 88);
            result = *(unsigned __int8 *)(v28 + 80);
          }
          if ( (_BYTE)result == 24 )
          {
            v28 = *(_QWORD *)(v28 + 88);
            result = *(unsigned __int8 *)(v28 + 80);
          }
        }
        if ( (_BYTE)result == 20 )
        {
          v13 = a1[7];
          result = sub_8B8060(v28, v13, *(_QWORD *)(v5 + 40), 1, 0, v14);
          if ( (_DWORD)result )
          {
            v13 = v28;
            result = sub_8B5FF0(&v52, v28, 0);
          }
        }
        if ( !v48 )
          break;
        v10 = *(_QWORD *)(v10 + 8);
        if ( !v10 )
          break;
        LODWORD(v12) = *(_DWORD *)(v5 + 16);
      }
      if ( v52 )
      {
        sub_893120(v52, 0, v54, v53, &v50, 0);
        v13 = a1[7];
        v33 = sub_8B7F20(v54[0], v13, *(_QWORD *)(v5 + 40), *(_BYTE *)(v5 + 18) & 1, 1, 0, 0, (__int64)v51);
        v34 = v50;
        v25 = v54[0];
        v10 = v33;
        v12 = 4 * (v51[0] & 1u);
        result = (unsigned int)v12 | a1[8] & 0xFB;
        *((_BYTE *)a1 + 64) = (4 * (v51[0] & 1)) | a1[8] & 0xFB;
        if ( v34 )
        {
          v13 = v5 + 8;
          result = sub_686890(872, v5 + 8, v10, a1[7], v35);
        }
        if ( v10 )
        {
          result = (__int64)&dword_4D04340;
          a1[1] = v10;
          v16 = dword_4D04340;
          if ( v47 == v45 )
            goto LABEL_53;
          v12 = *((unsigned __int8 *)a1 + 64);
          if ( (v12 & 1) == 0 )
            goto LABEL_29;
          v25 = v10;
        }
        else
        {
          v13 = v45;
          if ( v47 != v45 )
          {
            v12 = *((unsigned __int8 *)a1 + 64);
            if ( (v12 & 1) == 0 )
              goto LABEL_181;
          }
          if ( !v25 )
            goto LABEL_139;
          v16 = 0;
        }
        v10 = v25;
        goto LABEL_53;
      }
    }
    if ( v47 == v45 )
      goto LABEL_139;
    LOBYTE(v12) = *((_BYTE *)a1 + 64);
    goto LABEL_162;
  }
  v13 = v45;
  if ( v47 == v45 || (v12 = *((unsigned __int8 *)a1 + 64), (v12 & 1) != 0) )
  {
    v10 = v25;
    goto LABEL_52;
  }
LABEL_181:
  v10 = v25;
LABEL_28:
  v16 = 0;
LABEL_29:
  v13 = v42;
  if ( v42 )
  {
    a1[1] = v10;
    v17 = v10;
  }
  else
  {
    v17 = a1[1];
  }
  if ( v17 )
    goto LABEL_32;
  if ( v10 )
    goto LABEL_38;
  return result;
}
