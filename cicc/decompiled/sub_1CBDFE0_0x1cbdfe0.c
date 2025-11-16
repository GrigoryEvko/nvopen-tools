// Function: sub_1CBDFE0
// Address: 0x1cbdfe0
//
__int64 __fastcall sub_1CBDFE0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  int v6; // edx
  __int64 v7; // r12
  _QWORD *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r12
  _BOOL4 v12; // r15d
  __int64 v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r12
  __int64 v17; // r15
  _BOOL4 v18; // r13d
  _QWORD *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  int v22; // r12d
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r15
  int v26; // r15d
  __int64 v27; // rax
  __int64 v28; // rdx
  int v29; // eax
  int v30; // r8d
  __int64 v31; // rcx
  unsigned int v32; // r15d
  unsigned int v33; // esi
  char v34; // al
  int v35; // r12d
  int v36; // edx
  __int64 v37; // rcx
  _QWORD *v38; // rax
  __int64 v39; // rdx
  __int64 v40; // [rsp+8h] [rbp-58h]
  unsigned int v41; // [rsp+14h] [rbp-4Ch]
  unsigned __int64 v42[2]; // [rsp+18h] [rbp-48h] BYREF
  _QWORD v43[7]; // [rsp+28h] [rbp-38h] BYREF

  v6 = *(unsigned __int8 *)(a2 + 16);
  v42[0] = a4;
  if ( (unsigned int)(v6 - 35) <= 0x11 )
    return a2;
  if ( (_BYTE)v6 != 84 )
  {
    switch ( (_BYTE)v6 )
    {
      case 'W':
        if ( *(_QWORD *)(a2 - 24) != a3 )
          return 0;
LABEL_8:
        v9 = sub_1CBCEE0(a1 + 224, v42);
        v11 = v10;
        if ( v10 )
        {
          v12 = 1;
          if ( !v9 && v10 != a1 + 232 )
            v12 = v42[0] < *(_QWORD *)(v10 + 32);
          v13 = sub_22077B0(40);
          *(_QWORD *)(v13 + 32) = v42[0];
          sub_220F040(v12, v13, v11, a1 + 232);
          ++*(_QWORD *)(a1 + 264);
        }
        return 0;
      case '6':
        return 0;
      case '7':
        if ( *(_QWORD *)(a2 - 48) != a3 )
          return 0;
        v14 = sub_1CBCEE0(a1 + 224, v42);
        v16 = v15;
        if ( !v15 )
          return 0;
        v17 = a1 + 232;
        v18 = 1;
        if ( v14 || v15 == v17 )
          goto LABEL_25;
        goto LABEL_21;
      case ':':
        if ( a3 != *(_QWORD *)(a2 - 24) )
          return 0;
        v19 = sub_1CBCEE0(a1 + 224, v42);
        v16 = v20;
        if ( !v20 )
          return 0;
        v17 = a1 + 232;
        v18 = 1;
        if ( v19 )
          goto LABEL_25;
        if ( v20 == v17 )
        {
          v18 = 1;
          goto LABEL_25;
        }
LABEL_21:
        v18 = v42[0] < *(_QWORD *)(v16 + 32);
LABEL_25:
        v21 = sub_22077B0(40);
        *(_QWORD *)(v21 + 32) = v42[0];
        sub_220F040(v18, v21, v16, v17);
        ++*(_QWORD *)(a1 + 264);
        return 0;
      case '8':
        if ( a3 != *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)) )
          return 0;
        return a2;
    }
    if ( (unsigned int)(v6 - 60) <= 0xC || (_BYTE)v6 == 77 )
      return a2;
    if ( (_BYTE)v6 == 79 )
    {
      if ( a3 != *(_QWORD *)(a2 - 48) && a3 != *(_QWORD *)(a2 - 24) )
        return 0;
      return a2;
    }
    if ( (_BYTE)v6 != 78 )
    {
      if ( (_BYTE)v6 == 29 )
      {
        v35 = *(_DWORD *)(a2 + 20);
        v36 = (v35 & 0xFFFFFFF) - 3 - sub_154CBE0(a2);
        if ( !v36 )
          return 0;
        v37 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
        v38 = (_QWORD *)(a2 - 24 * v37);
        v39 = a2 + 24 * ((unsigned int)(v36 - 1) - v37) + 24;
        while ( a3 != *v38 )
        {
          v38 += 3;
          if ( (_QWORD *)v39 == v38 )
            return 0;
        }
      }
LABEL_56:
      sub_1CBCF80(a1 + 224, v42);
      return 0;
    }
    v22 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    if ( *(char *)(a2 + 23) < 0 )
    {
      v23 = sub_1648A40(a2);
      v25 = v23 + v24;
      if ( *(char *)(a2 + 23) >= 0 )
      {
        if ( (unsigned int)(v25 >> 4) )
          goto LABEL_64;
      }
      else if ( (unsigned int)((v25 - sub_1648A40(a2)) >> 4) )
      {
        if ( *(char *)(a2 + 23) < 0 )
        {
          v26 = *(_DWORD *)(sub_1648A40(a2) + 8);
          if ( *(char *)(a2 + 23) >= 0 )
            BUG();
          v27 = sub_1648A40(a2);
          v29 = *(_DWORD *)(v27 + v28 - 4) - v26;
LABEL_41:
          v30 = v22 - 1;
          v7 = *(_QWORD *)(a2 - 24);
          if ( *(_BYTE *)(v7 + 16) )
            v7 = 0;
          v31 = 0;
          v32 = v30 - v29;
          if ( v30 != v29 )
          {
            while ( 1 )
            {
              v33 = v31 + 1;
              if ( a3 == *(_QWORD *)(a2 + 24 * (v31 - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) )
              {
                if ( !v7 )
                {
                  sub_1CBCF80(a1 + 224, v42);
                  return v7;
                }
                if ( (unsigned __int64)v32 > *(_QWORD *)(v7 + 96) )
                  goto LABEL_56;
                v40 = v31;
                v41 = v31 + 1;
                v43[0] = *(_QWORD *)(v7 + 112);
                v34 = sub_1560260(v43, (int)v31 + 1, 22);
                v33 = v41;
                v31 = v40;
                if ( v34 )
                  return a2;
              }
              ++v31;
              if ( v32 <= v33 )
                goto LABEL_56;
            }
          }
          goto LABEL_56;
        }
LABEL_64:
        BUG();
      }
    }
    v29 = 0;
    goto LABEL_41;
  }
  if ( *(_QWORD *)(a2 - 48) == a3 )
    goto LABEL_8;
  return 0;
}
