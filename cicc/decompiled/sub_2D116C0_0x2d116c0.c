// Function: sub_2D116C0
// Address: 0x2d116c0
//
unsigned __int8 *__fastcall sub_2D116C0(__int64 a1, unsigned __int8 *a2, __int64 a3, unsigned __int64 a4)
{
  int v5; // edx
  __int64 v6; // rcx
  __int64 v7; // r12
  _QWORD *v9; // rax
  _QWORD *v10; // rdx
  _QWORD *v11; // r12
  char v12; // r15
  __int64 v13; // rax
  _QWORD *v14; // rax
  _QWORD *v15; // rdx
  _QWORD *v16; // r12
  _QWORD *v17; // r15
  char v18; // r14
  _QWORD *v19; // rax
  _QWORD *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r12
  int v25; // r12d
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // r15
  __int64 v31; // r14
  int v32; // edx
  __int64 v33; // rcx
  __int64 v34; // rdx
  unsigned __int8 *v35; // rax
  unsigned __int64 v36[7]; // [rsp+8h] [rbp-38h] BYREF

  v5 = *a2;
  v36[0] = a4;
  v6 = (unsigned int)(v5 - 42);
  if ( (unsigned int)v6 <= 0x11 )
    return a2;
  if ( (_BYTE)v5 != 91 )
  {
    switch ( (_BYTE)v5 )
    {
      case '^':
        if ( *((_QWORD *)a2 - 4) != a3 )
          return 0;
LABEL_8:
        v9 = sub_2D11450(a1 + 224, v36);
        v11 = v10;
        if ( v10 )
        {
          v12 = 1;
          if ( !v9 && v10 != (_QWORD *)(a1 + 232) )
            v12 = v36[0] < v10[4];
          v13 = sub_22077B0(0x28u);
          *(_QWORD *)(v13 + 32) = v36[0];
          sub_220F040(v12, v13, v11, (_QWORD *)(a1 + 232));
          ++*(_QWORD *)(a1 + 264);
        }
        return 0;
      case '=':
        return 0;
      case '>':
        if ( *((_QWORD *)a2 - 8) != a3 )
          return 0;
        v14 = sub_2D11450(a1 + 224, v36);
        v16 = v15;
        if ( !v15 )
          return 0;
        v17 = (_QWORD *)(a1 + 232);
        v18 = 1;
        if ( v14 || v15 == v17 )
          goto LABEL_25;
        goto LABEL_21;
      case 'A':
        if ( a3 != *((_QWORD *)a2 - 4) )
          return 0;
        v19 = sub_2D11450(a1 + 224, v36);
        v16 = v20;
        if ( !v20 )
          return 0;
        v17 = (_QWORD *)(a1 + 232);
        v18 = 1;
        if ( v19 )
          goto LABEL_25;
        if ( v20 == v17 )
        {
          v18 = 1;
          goto LABEL_25;
        }
LABEL_21:
        v18 = v36[0] < v16[4];
LABEL_25:
        v21 = sub_22077B0(0x28u);
        *(_QWORD *)(v21 + 32) = v36[0];
        sub_220F040(v18, v21, v16, v17);
        ++*(_QWORD *)(a1 + 264);
        return 0;
      case '?':
        if ( a3 != *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)] )
          return 0;
        return a2;
    }
    if ( (unsigned int)(v5 - 67) <= 0xC || (_BYTE)v5 == 84 )
      return a2;
    if ( (_BYTE)v5 == 86 )
    {
      if ( a3 != *((_QWORD *)a2 - 8) && a3 != *((_QWORD *)a2 - 4) )
        return 0;
      return a2;
    }
    if ( (_BYTE)v5 != 85 )
    {
      if ( (_BYTE)v5 == 34 )
      {
        v32 = sub_A17190(a2);
        if ( !v32 )
          return 0;
        v33 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
        v34 = (__int64)&a2[32 * ((unsigned int)(v32 - 1) - v33) + 32];
        v35 = &a2[-32 * v33];
        while ( a3 != *(_QWORD *)v35 )
        {
          v35 += 32;
          if ( v35 == (unsigned __int8 *)v34 )
            return 0;
        }
      }
LABEL_61:
      sub_2D114F0(a1 + 224, v36);
      return 0;
    }
    if ( (a2[7] & 0x80u) != 0 )
    {
      v22 = sub_BD2BC0((__int64)a2);
      v24 = v22 + v23;
      if ( (a2[7] & 0x80u) == 0 )
      {
        if ( (unsigned int)(v24 >> 4) )
          goto LABEL_68;
      }
      else if ( (unsigned int)((v24 - sub_BD2BC0((__int64)a2)) >> 4) )
      {
        if ( (a2[7] & 0x80u) != 0 )
        {
          v25 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
          if ( (a2[7] & 0x80u) == 0 )
            BUG();
          v26 = sub_BD2BC0((__int64)a2);
          v28 = (unsigned int)(*(_DWORD *)(v26 + v27 - 4) - v25);
LABEL_41:
          v7 = *((_QWORD *)a2 - 4);
          v29 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
          v30 = (32 * v29 - 32 - 32 * v28) >> 5;
          if ( v7 )
          {
            if ( *(_BYTE *)v7 )
            {
              v7 = 0;
            }
            else
            {
              v6 = *((_QWORD *)a2 + 10);
              if ( *(_QWORD *)(v7 + 24) != v6 )
                v7 = 0;
            }
          }
          if ( (_DWORD)v30 )
          {
            v31 = 0;
            while ( 1 )
            {
              if ( a3 == *(_QWORD *)&a2[32 * (v31 - v29)] )
              {
                if ( !v7 )
                {
                  sub_2D114F0(a1 + 224, v36);
                  return (unsigned __int8 *)v7;
                }
                if ( (unsigned __int64)(unsigned int)v30 > *(_QWORD *)(v7 + 104) )
                  goto LABEL_61;
                if ( (*(_BYTE *)(v7 + 2) & 1) != 0 )
                  sub_B2C6D0(v7, (__int64)a2, v31 - v29, v6);
                if ( sub_B2BE10(*(_QWORD *)(v7 + 96) + 40 * v31) )
                  return a2;
              }
              if ( (unsigned int)v30 == ++v31 )
                goto LABEL_61;
              v29 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
            }
          }
          goto LABEL_61;
        }
LABEL_68:
        BUG();
      }
    }
    v28 = 0;
    goto LABEL_41;
  }
  if ( *((_QWORD *)a2 - 8) == a3 )
    goto LABEL_8;
  return 0;
}
