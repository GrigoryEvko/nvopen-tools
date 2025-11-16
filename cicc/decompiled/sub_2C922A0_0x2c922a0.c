// Function: sub_2C922A0
// Address: 0x2c922a0
//
char __fastcall sub_2C922A0(__int64 a1, __int64 a2, __int64 *a3, unsigned __int64 a4, unsigned __int64 *a5)
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
  __int64 v28; // rsi
  __int64 v29; // rdx
  __int64 *v30; // rax
  __int64 v31; // rsi
  __int64 v32; // r15
  __int64 v33; // rdx
  unsigned int v34; // r11d
  __int64 v35; // rdx
  unsigned int v36; // r11d
  __int64 v37; // rdi
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // rax
  unsigned __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // r15
  unsigned __int64 v43; // rax
  unsigned int v44; // eax
  char v45; // cl
  __int64 v46; // rdx
  int v47; // edx
  __int64 v48; // rdi
  __int64 v49; // rax
  unsigned int v50; // eax
  int v51; // [rsp+8h] [rbp-58h]
  __int64 v52; // [rsp+8h] [rbp-58h]
  unsigned int v53; // [rsp+10h] [rbp-50h]
  __int64 v54; // [rsp+18h] [rbp-48h]
  __int64 v55; // [rsp+18h] [rbp-48h]
  _QWORD v56[7]; // [rsp+28h] [rbp-38h] BYREF

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
          v15 = 0;
          if ( v11 )
            v15 = (__int64)(v12 << (64 - (unsigned __int8)v11)) >> (64 - (unsigned __int8)v11);
          goto LABEL_7;
        }
        goto LABEL_24;
      }
      v18 = *(_QWORD *)v12;
      v15 = *(_QWORD *)v12;
      if ( (*(_QWORD *)(v12 + 8LL * (v13 >> 6)) & v14) != 0 )
      {
LABEL_7:
        v16 = v15 + *a5;
        *a5 = v16;
        return v16 <= a4;
      }
    }
    else
    {
      if ( v11 <= 0x40 )
      {
LABEL_24:
        v19 = v12;
        goto LABEL_11;
      }
      v18 = *(_QWORD *)v12;
    }
    v19 = v18;
LABEL_11:
    *a3 = v19;
    return a4 >= v19;
  }
  v20 = a2;
  while ( 1 )
  {
    if ( v9 == 15 )
    {
      if ( (unsigned int)sub_2C91F50(*(char **)(v7 - 8), a3) )
        return a4 >= *a3;
      v27 = *(_QWORD *)(v7 - 8);
      if ( *(_BYTE *)v27 == 85 )
      {
        v46 = *(_QWORD *)(v27 - 32);
        if ( v46 )
        {
          if ( !*(_BYTE *)v46 && *(_QWORD *)(v46 + 24) == *(_QWORD *)(v27 + 80) && (*(_BYTE *)(v46 + 33) & 0x20) != 0 )
          {
            v47 = *(_DWORD *)(v46 + 36);
            if ( (unsigned int)(v47 - 9370) <= 2 )
            {
              v40 = (int)qword_5012588 - 1;
              *a3 = v40;
              return a4 >= v40;
            }
            if ( (unsigned int)(v47 - 9360) <= 2 || v47 == 9374 )
            {
              v40 = (int)qword_5012588;
              *a3 = (int)qword_5012588;
              return a4 >= v40;
            }
            if ( v47 == 9307 )
            {
              *a3 = 2147483646;
              v40 = 2147483646;
              return a4 >= v40;
            }
            if ( (unsigned int)(v47 - 9308) <= 1 )
            {
              *a3 = 65534;
              v40 = 65534;
              return a4 >= v40;
            }
            if ( v47 == 9355 )
            {
              *a3 = 0x7FFFFFFF;
              v40 = 0x7FFFFFFF;
              return a4 >= v40;
            }
            if ( (unsigned int)(v47 - 9356) <= 1 )
            {
              *a3 = 0xFFFF;
              v40 = 0xFFFF;
              return a4 >= v40;
            }
          }
        }
      }
      v28 = *(_QWORD *)(v27 + 8);
      if ( *(_BYTE *)(v28 + 8) == 12 )
      {
        v44 = sub_D97050(v21, v28);
        v45 = v44;
        if ( v44 <= 0x1F )
          goto LABEL_61;
      }
      return 0;
    }
    if ( v9 == 7 )
      break;
    if ( v9 == 6 )
    {
      if ( (*(_BYTE *)(v7 + 28) & 2) != 0 )
        goto LABEL_63;
      if ( a5 )
      {
        v29 = *(_QWORD *)(v7 + 40);
        if ( (_DWORD)v29 )
        {
          v30 = *(__int64 **)(v7 + 32);
          v31 = (__int64)&v30[(unsigned int)(v29 - 1) + 1];
          while ( 1 )
          {
            v32 = *v30;
            if ( !*(_WORD *)(*v30 + 24) )
            {
              v33 = *(_QWORD *)(v32 + 32);
              v34 = *(_DWORD *)(v33 + 32);
              v35 = *(_QWORD *)(v33 + 24);
              if ( v34 > 0x40 )
                v35 = *(_QWORD *)(v35 + 8LL * ((v34 - 1) >> 6));
              if ( (v35 & (1LL << ((unsigned __int8)v34 - 1))) != 0 )
                break;
            }
            if ( (__int64 *)v31 == ++v30 )
              goto LABEL_40;
          }
          *a3 = 1;
          v51 = *(_QWORD *)(v7 + 40);
          if ( !v51 )
          {
LABEL_49:
            v39 = *a3
                * -sub_2C90D50(*(_QWORD *)(*(_QWORD *)(v32 + 32) + 24LL), *(_DWORD *)(*(_QWORD *)(v32 + 32) + 32LL));
            if ( a4 < v39 )
              return 0;
            v40 = *a5 + v39;
            *a5 = v40;
            return a4 >= v40;
          }
        }
        else
        {
LABEL_40:
          *a3 = 1;
          v51 = *(_QWORD *)(v7 + 40);
          if ( !v51 )
            return 1;
          v32 = 0;
        }
      }
      else
      {
        *a3 = 1;
        v32 = 0;
        v51 = *(_QWORD *)(v7 + 40);
        if ( !v51 )
          return 1;
      }
      v36 = 0;
      while ( 1 )
      {
        v37 = *(_QWORD *)(*(_QWORD *)(v7 + 32) + 8LL * v36);
        if ( v37 != v32 )
        {
          v53 = v36;
          v54 = v20;
          if ( !(unsigned __int8)sub_2C922A0(v37, v20, v56, a4, 0) )
            return 0;
          if ( a4 < v56[0] )
            return 0;
          v38 = *a3 * v56[0];
          v20 = v54;
          v36 = v53;
          *a3 = v38;
          if ( a4 < v38 )
            return 0;
        }
        if ( ++v36 == v51 )
        {
          if ( !v32 )
            return 1;
          goto LABEL_49;
        }
      }
    }
    if ( v9 == 5 )
    {
      if ( (*(_BYTE *)(v7 + 28) & 2) == 0 )
      {
        *a3 = 0;
        v41 = *(_QWORD *)(v7 + 40);
        if ( !(_DWORD)v41 )
          return 1;
        v42 = 0;
        v52 = 8LL * (unsigned int)v41;
        while ( 1 )
        {
          v55 = v20;
          if ( !(unsigned __int8)sub_2C922A0(*(_QWORD *)(*(_QWORD *)(v7 + 32) + v42), v20, v56, a4, a5) )
            break;
          v43 = *a3 + v56[0];
          *a3 = v43;
          if ( a4 < v43 )
            break;
          v42 += 8;
          v20 = v55;
          if ( v52 == v42 )
            return 1;
        }
        return 0;
      }
LABEL_63:
      *a3 = a4;
      return 1;
    }
    if ( v9 != 8 )
    {
      if ( v9 == 3 )
      {
        v48 = *(_QWORD *)(v7 + 32);
        goto LABEL_89;
      }
      if ( v9 == 2 )
      {
        v48 = v7;
LABEL_89:
        v49 = sub_D95540(v48);
        v50 = sub_D97050(a2, v49);
        v45 = v50;
        if ( v50 > 0x20 )
          return 0;
LABEL_61:
        *a3 = (1LL << v45) - 1;
        return 1;
      }
      return 0;
    }
    if ( *(_QWORD *)(v7 + 40) != 2 || (*(_BYTE *)(v7 + 28) & 2) == 0 )
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
    v25 = (unsigned __int8)sub_2C922A0(*(_QWORD *)(v7 + 32), a2, v56, a4, 0) == 0;
    v26 = a4;
    if ( !v25 )
      v26 = v56[0];
    v19 = v26 / v24;
    goto LABEL_11;
  }
  return result;
}
