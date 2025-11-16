// Function: sub_1C52500
// Address: 0x1c52500
//
char __fastcall sub_1C52500(__int64 a1, __int64 a2, __int64 *a3, unsigned __int64 a4, unsigned __int64 *a5)
{
  __int64 v7; // r12
  __int16 v9; // ax
  __int64 v10; // rax
  unsigned int v11; // esi
  unsigned __int64 v12; // rdi
  unsigned int v13; // ecx
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  char result; // al
  __int64 v18; // rsi
  unsigned __int64 v19; // rax
  __int64 v20; // r9
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned __int64 v24; // rbx
  bool v25; // zf
  unsigned __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 *v29; // rax
  __int64 v30; // rsi
  __int64 v31; // r15
  __int64 v32; // rdx
  unsigned int v33; // r11d
  __int64 v34; // rdx
  unsigned int v35; // r11d
  __int64 v36; // rdi
  unsigned __int64 v37; // rax
  __int64 v38; // rax
  unsigned int v39; // edx
  __int64 *v40; // rax
  __int64 v41; // rax
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // r15
  unsigned __int64 v46; // rax
  unsigned int v47; // eax
  char v48; // cl
  __int64 v49; // rdx
  int v50; // edx
  __int64 v51; // rdi
  __int64 v52; // rax
  unsigned int v53; // eax
  int v54; // [rsp+8h] [rbp-58h]
  __int64 v55; // [rsp+8h] [rbp-58h]
  unsigned int v56; // [rsp+10h] [rbp-50h]
  __int64 v57; // [rsp+18h] [rbp-48h]
  __int64 v58; // [rsp+18h] [rbp-48h]
  _QWORD v59[7]; // [rsp+28h] [rbp-38h] BYREF

  v7 = a1;
  v9 = *(_WORD *)(a1 + 24);
  if ( !v9 )
  {
LABEL_2:
    v10 = *(_QWORD *)(v7 + 32);
    v11 = *(_DWORD *)(v10 + 32);
    v12 = *(_QWORD *)(v10 + 24);
    if ( a5 )
    {
      v13 = v11 - 1;
      v14 = 1LL << ((unsigned __int8)v11 - 1);
      if ( v11 <= 0x40 )
      {
        if ( (v14 & v12) != 0 )
        {
          v15 = (__int64)(v12 << (64 - (unsigned __int8)v11)) >> (64 - (unsigned __int8)v11);
LABEL_6:
          v16 = v15 + *a5;
          *a5 = v16;
          return v16 <= a4;
        }
        goto LABEL_23;
      }
      v18 = *(_QWORD *)v12;
      v15 = *(_QWORD *)v12;
      if ( (*(_QWORD *)(v12 + 8LL * (v13 >> 6)) & v14) != 0 )
        goto LABEL_6;
    }
    else
    {
      if ( v11 <= 0x40 )
      {
LABEL_23:
        v19 = v12;
        goto LABEL_10;
      }
      v18 = *(_QWORD *)v12;
    }
    v19 = v18;
LABEL_10:
    *a3 = v19;
    return a4 >= v19;
  }
  v20 = a2;
  while ( 1 )
  {
    if ( v9 == 10 )
    {
      if ( (unsigned int)sub_1C521F0(*(_QWORD *)(v7 - 8), a3) )
        return a4 >= *a3;
      v27 = *(_QWORD *)(v7 - 8);
      if ( *(_BYTE *)(v27 + 16) == 78 )
      {
        v49 = *(_QWORD *)(v27 - 24);
        if ( !*(_BYTE *)(v49 + 16) && (*(_BYTE *)(v49 + 33) & 0x20) != 0 )
        {
          v50 = *(_DWORD *)(v49 + 36);
          if ( v50 == 3778 )
          {
            *a3 = 32;
            v43 = 32;
            return a4 >= v43;
          }
          if ( v50 == 3779 )
          {
            *a3 = 64;
            v43 = 64;
            return a4 >= v43;
          }
          if ( (unsigned int)(v50 - 4344) <= 2 )
          {
            v43 = dword_4FBCA00 - 1;
            *a3 = v43;
            return a4 >= v43;
          }
          if ( (unsigned int)(v50 - 4334) <= 2 || v50 == 4348 )
          {
            v43 = dword_4FBCA00;
            *a3 = dword_4FBCA00;
            return a4 >= v43;
          }
          if ( v50 == 4286 )
          {
            *a3 = 2147483646;
            v43 = 2147483646;
            return a4 >= v43;
          }
          if ( (unsigned int)(v50 - 4287) <= 1 )
          {
            *a3 = 65534;
            v43 = 65534;
            return a4 >= v43;
          }
          if ( v50 == 4329 )
          {
            *a3 = 0x7FFFFFFF;
            v43 = 0x7FFFFFFF;
            return a4 >= v43;
          }
          if ( (unsigned int)(v50 - 4330) <= 1 )
          {
            *a3 = 0xFFFF;
            v43 = 0xFFFF;
            return a4 >= v43;
          }
        }
      }
      if ( *(_BYTE *)(*(_QWORD *)v27 + 8LL) == 11 )
      {
        v47 = sub_1456C90(v21, *(_QWORD *)v27);
        v48 = v47;
        if ( v47 <= 0x1F )
          goto LABEL_62;
      }
      return 0;
    }
    if ( v9 == 6 )
      break;
    if ( v9 == 5 )
    {
      if ( (*(_BYTE *)(v7 + 26) & 2) != 0 )
        goto LABEL_64;
      if ( a5 )
      {
        v28 = *(_QWORD *)(v7 + 40);
        if ( (_DWORD)v28 )
        {
          v29 = *(__int64 **)(v7 + 32);
          v30 = (__int64)&v29[(unsigned int)(v28 - 1) + 1];
          while ( 1 )
          {
            v31 = *v29;
            if ( !*(_WORD *)(*v29 + 24) )
            {
              v32 = *(_QWORD *)(v31 + 32);
              v33 = *(_DWORD *)(v32 + 32);
              v34 = *(_QWORD *)(v32 + 24);
              if ( v33 > 0x40 )
                v34 = *(_QWORD *)(v34 + 8LL * ((v33 - 1) >> 6));
              if ( (v34 & (1LL << ((unsigned __int8)v33 - 1))) != 0 )
                break;
            }
            if ( (__int64 *)v30 == ++v29 )
              goto LABEL_39;
          }
          *a3 = 1;
          v54 = *(_QWORD *)(v7 + 40);
          if ( !v54 )
          {
LABEL_48:
            v38 = *(_QWORD *)(v31 + 32);
            v39 = *(_DWORD *)(v38 + 32);
            v40 = *(__int64 **)(v38 + 24);
            if ( v39 > 0x40 )
              v41 = *v40;
            else
              v41 = (__int64)((_QWORD)v40 << (64 - (unsigned __int8)v39)) >> (64 - (unsigned __int8)v39);
            v42 = *a3 * -v41;
            if ( a4 < v42 )
              return 0;
            v43 = *a5 + v42;
            *a5 = v43;
            return a4 >= v43;
          }
        }
        else
        {
LABEL_39:
          *a3 = 1;
          v54 = *(_QWORD *)(v7 + 40);
          if ( !v54 )
            return 1;
          v31 = 0;
        }
      }
      else
      {
        *a3 = 1;
        v31 = 0;
        v54 = *(_QWORD *)(v7 + 40);
        if ( !v54 )
          return 1;
      }
      v35 = 0;
      while ( 1 )
      {
        v36 = *(_QWORD *)(*(_QWORD *)(v7 + 32) + 8LL * v35);
        if ( v36 != v31 )
        {
          v56 = v35;
          v57 = v20;
          if ( !(unsigned __int8)sub_1C52500(v36, v20, v59, a4, 0) )
            return 0;
          if ( a4 < v59[0] )
            return 0;
          v37 = *a3 * v59[0];
          v20 = v57;
          v35 = v56;
          *a3 = v37;
          if ( a4 < v37 )
            return 0;
        }
        if ( ++v35 == v54 )
        {
          if ( !v31 )
            return 1;
          goto LABEL_48;
        }
      }
    }
    if ( v9 == 4 )
    {
      if ( (*(_BYTE *)(v7 + 26) & 2) == 0 )
      {
        *a3 = 0;
        v44 = *(_QWORD *)(v7 + 40);
        if ( !(_DWORD)v44 )
          return 1;
        v45 = 0;
        v55 = 8LL * (unsigned int)v44;
        while ( 1 )
        {
          v58 = v20;
          if ( !(unsigned __int8)sub_1C52500(*(_QWORD *)(*(_QWORD *)(v7 + 32) + v45), v20, v59, a4, a5) )
            break;
          v46 = *a3 + v59[0];
          *a3 = v46;
          if ( a4 < v46 )
            break;
          v45 += 8;
          v20 = v58;
          if ( v55 == v45 )
            return 1;
        }
        return 0;
      }
LABEL_64:
      *a3 = a4;
      return 1;
    }
    if ( v9 != 7 )
    {
      if ( v9 == 2 )
      {
        v51 = *(_QWORD *)(v7 + 32);
        goto LABEL_92;
      }
      if ( v9 == 1 )
      {
        v51 = v7;
LABEL_92:
        v52 = sub_1456040(v51);
        v53 = sub_1456C90(a2, v52);
        v48 = v53;
        if ( v53 > 0x20 )
          return 0;
LABEL_62:
        *a3 = (1LL << v48) - 1;
        return 1;
      }
      return 0;
    }
    if ( *(_QWORD *)(v7 + 40) != 2 || (*(_BYTE *)(v7 + 26) & 2) == 0 )
      return 0;
    v7 = **(_QWORD **)(v7 + 32);
    v9 = *(_WORD *)(v7 + 24);
    if ( !v9 )
      goto LABEL_2;
  }
  v22 = *(_QWORD *)(v7 + 40);
  if ( *(_WORD *)(v22 + 24) )
    return 0;
  v23 = *(_QWORD *)(v22 + 32);
  v24 = *(_QWORD *)(v23 + 24);
  if ( *(_DWORD *)(v23 + 32) > 0x40u )
    v24 = *(_QWORD *)v24;
  result = 0;
  if ( v24 )
  {
    v25 = (unsigned __int8)sub_1C52500(*(_QWORD *)(v7 + 32), a2, v59, a4, 0) == 0;
    v26 = a4;
    if ( !v25 )
      v26 = v59[0];
    v19 = v26 / v24;
    goto LABEL_10;
  }
  return result;
}
