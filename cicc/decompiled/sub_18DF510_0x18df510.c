// Function: sub_18DF510
// Address: 0x18df510
//
__int64 __fastcall sub_18DF510(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  unsigned int v9; // esi
  __int64 v10; // r8
  __int64 v11; // rdi
  unsigned int v12; // ecx
  __int64 result; // rax
  __int64 v14; // rdx
  __int64 v15; // r13
  int v16; // eax
  int v17; // edx
  __int64 v18; // rax
  __int64 v19; // r14
  bool v20; // zf
  int v21; // eax
  int v22; // ecx
  __int64 v23; // rdi
  unsigned int v24; // eax
  __int64 v25; // rsi
  __int64 v26; // r8
  __int64 v27; // r13
  __int64 v28; // r12
  __int64 v29; // r12
  unsigned int v30; // r15d
  __int64 v31; // rcx
  _QWORD *v32; // rdi
  _QWORD *v33; // rsi
  __int64 v34; // rdx
  __int64 v35; // rsi
  __int64 v36; // rdx
  int v37; // eax
  int v38; // eax
  __int64 v39; // rsi
  __int64 v40; // rdi
  unsigned int v41; // r14d
  __int64 v42; // rcx
  __int64 *v43; // rdi
  unsigned int v44; // r10d
  _QWORD *v45; // rsi
  __int64 v46; // [rsp+0h] [rbp-70h]
  __int64 v47; // [rsp+10h] [rbp-60h]
  int v48; // [rsp+1Ch] [rbp-54h]
  const void *v49; // [rsp+20h] [rbp-50h]
  __int64 v50; // [rsp+28h] [rbp-48h]
  __int64 v51[7]; // [rsp+38h] [rbp-38h] BYREF

  v6 = a1 + 80;
  v9 = *(_DWORD *)(a1 + 104);
  v49 = (const void *)(a1 + 128);
  v50 = a1 + 1616;
  if ( !v9 )
    goto LABEL_23;
LABEL_2:
  LODWORD(v10) = v9 - 1;
  v11 = *(_QWORD *)(a1 + 88);
  v12 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = v11 + 24LL * v12;
  v14 = *(_QWORD *)result;
  if ( *(_QWORD *)result != a2 )
  {
    LODWORD(a6) = 1;
    v15 = 0;
    while ( v14 != -8 )
    {
      if ( v14 == -16 && !v15 )
        v15 = result;
      v12 = v10 & (a6 + v12);
      result = v11 + 24LL * v12;
      v14 = *(_QWORD *)result;
      if ( *(_QWORD *)result == a2 )
        goto LABEL_3;
      LODWORD(a6) = a6 + 1;
    }
    if ( !v15 )
      v15 = result;
    v16 = *(_DWORD *)(a1 + 96);
    ++*(_QWORD *)(a1 + 80);
    v17 = v16 + 1;
    if ( 4 * (v16 + 1) < 3 * v9 )
    {
      if ( v9 - *(_DWORD *)(a1 + 100) - v17 > v9 >> 3 )
        goto LABEL_11;
      sub_18DEA70(v6, v9);
      v37 = *(_DWORD *)(a1 + 104);
      if ( v37 )
      {
        v38 = v37 - 1;
        v39 = *(_QWORD *)(a1 + 88);
        LODWORD(v10) = 1;
        v40 = 0;
        v41 = v38 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v15 = v39 + 24LL * v41;
        v42 = *(_QWORD *)v15;
        v17 = *(_DWORD *)(a1 + 96) + 1;
        if ( *(_QWORD *)v15 != a2 )
        {
          while ( v42 != -8 )
          {
            if ( v42 == -16 && !v40 )
              v40 = v15;
            LODWORD(a6) = v10 + 1;
            v41 = v38 & (v10 + v41);
            v15 = v39 + 24LL * v41;
            v42 = *(_QWORD *)v15;
            if ( *(_QWORD *)v15 == a2 )
              goto LABEL_11;
            LODWORD(v10) = v10 + 1;
          }
          if ( v40 )
            v15 = v40;
        }
        goto LABEL_11;
      }
LABEL_101:
      ++*(_DWORD *)(a1 + 96);
      BUG();
    }
    while ( 1 )
    {
      sub_18DEA70(v6, 2 * v9);
      v21 = *(_DWORD *)(a1 + 104);
      if ( !v21 )
        goto LABEL_101;
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a1 + 88);
      v24 = (v21 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = v23 + 24LL * v24;
      v25 = *(_QWORD *)v15;
      v17 = *(_DWORD *)(a1 + 96) + 1;
      if ( *(_QWORD *)v15 != a2 )
      {
        LODWORD(a6) = 1;
        v10 = 0;
        while ( v25 != -8 )
        {
          if ( !v10 && v25 == -16 )
            v10 = v15;
          v24 = v22 & (a6 + v24);
          v15 = v23 + 24LL * v24;
          v25 = *(_QWORD *)v15;
          if ( *(_QWORD *)v15 == a2 )
            goto LABEL_11;
          LODWORD(a6) = a6 + 1;
        }
        if ( v10 )
          v15 = v10;
      }
LABEL_11:
      *(_DWORD *)(a1 + 96) = v17;
      if ( *(_QWORD *)v15 != -8 )
        --*(_DWORD *)(a1 + 100);
      *(_QWORD *)v15 = a2;
      *(_BYTE *)(v15 + 8) = 0;
      *(_QWORD *)(v15 + 16) = 0;
LABEL_14:
      *(_BYTE *)(v15 + 8) = 1;
      v18 = *(unsigned int *)(a1 + 120);
      if ( (unsigned int)v18 >= *(_DWORD *)(a1 + 124) )
      {
        sub_16CD150(a1 + 112, v49, 0, 8, v10, a6);
        v18 = *(unsigned int *)(a1 + 120);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 112) + 8 * v18) = a2;
      ++*(_DWORD *)(a1 + 120);
      result = sub_15C70A0(a2 + 48);
      if ( result )
        result = sub_18DEC30(a1, result);
      v19 = *(_QWORD *)(v15 + 16);
      if ( *(_QWORD *)(v19 + 24) == a2 )
        break;
LABEL_19:
      if ( *(_BYTE *)v19 )
        return result;
      v20 = *(_BYTE *)(v19 + 3) == 0;
      *(_BYTE *)v19 = 1;
      if ( v20 )
      {
        *(_BYTE *)(v19 + 3) = 1;
        v26 = *(_QWORD *)(v19 + 16);
        result = *(_QWORD *)(a1 + 1624);
        if ( *(_QWORD *)(a1 + 1632) != result )
          goto LABEL_32;
        v32 = (_QWORD *)(result + 8LL * *(unsigned int *)(a1 + 1644));
        LODWORD(a6) = *(_DWORD *)(a1 + 1644);
        if ( (_QWORD *)result == v32 )
        {
LABEL_80:
          if ( (unsigned int)a6 >= *(_DWORD *)(a1 + 1640) )
          {
LABEL_32:
            result = (__int64)sub_16CCBA0(v50, *(_QWORD *)(v19 + 16));
            goto LABEL_21;
          }
          LODWORD(a6) = a6 + 1;
          *(_DWORD *)(a1 + 1644) = a6;
          *v32 = v26;
          ++*(_QWORD *)(a1 + 1616);
        }
        else
        {
          v33 = 0;
          while ( v26 != *(_QWORD *)result )
          {
            if ( *(_QWORD *)result == -2 )
              v33 = (_QWORD *)result;
            result += 8;
            if ( v32 == (_QWORD *)result )
            {
              if ( !v33 )
                goto LABEL_80;
              *v33 = v26;
              --*(_DWORD *)(a1 + 1648);
              ++*(_QWORD *)(a1 + 1616);
              break;
            }
          }
        }
      }
LABEL_21:
      if ( !*(_BYTE *)(v19 + 1) )
        return result;
      v9 = *(_DWORD *)(a1 + 104);
      a2 = *(_QWORD *)(v19 + 24);
      if ( v9 )
        goto LABEL_2;
LABEL_23:
      ++*(_QWORD *)(a1 + 80);
    }
    v27 = *(_QWORD *)(v19 + 16);
    result = *(_QWORD *)(a1 + 1456);
    if ( *(_QWORD *)(a1 + 1464) == result )
    {
      v36 = result + 8LL * *(unsigned int *)(a1 + 1476);
      if ( result == v36 )
      {
LABEL_69:
        result = v36;
      }
      else
      {
        while ( v27 != *(_QWORD *)result )
        {
          result += 8;
          if ( v36 == result )
            goto LABEL_69;
        }
      }
    }
    else
    {
      result = (__int64)sub_16CC9F0(a1 + 1448, *(_QWORD *)(v19 + 16));
      if ( v27 == *(_QWORD *)result )
      {
        v34 = *(_QWORD *)(a1 + 1464);
        if ( v34 == *(_QWORD *)(a1 + 1456) )
          v35 = *(unsigned int *)(a1 + 1476);
        else
          v35 = *(unsigned int *)(a1 + 1472);
        v36 = v34 + 8 * v35;
      }
      else
      {
        result = *(_QWORD *)(a1 + 1464);
        if ( result != *(_QWORD *)(a1 + 1456) )
        {
LABEL_36:
          if ( *(_BYTE *)(v19 + 1) )
            goto LABEL_19;
          v28 = *(_QWORD *)(a2 + 40);
          result = sub_157EBA0(v28);
          if ( !result )
            goto LABEL_19;
          v48 = sub_15F4D60(result);
          result = sub_157EBA0(v28);
          v29 = result;
          if ( !v48 )
            goto LABEL_19;
          v46 = v6;
          v30 = 0;
          while ( 1 )
          {
            v51[0] = sub_15F4DF0(v29, v30);
            result = sub_18DF0D0(a1 + 24, v51);
            v31 = result;
            if ( !*(_BYTE *)result )
              break;
LABEL_42:
            if ( ++v30 == v48 )
            {
              v6 = v46;
              goto LABEL_19;
            }
          }
          v20 = *(_BYTE *)(result + 3) == 0;
          *(_BYTE *)result = 1;
          if ( v20 )
          {
            *(_BYTE *)(result + 3) = 1;
            a6 = *(_QWORD *)(result + 16);
            result = *(_QWORD *)(a1 + 1624);
            if ( *(_QWORD *)(a1 + 1632) != result )
              goto LABEL_46;
            v43 = (__int64 *)(result + 8LL * *(unsigned int *)(a1 + 1644));
            v44 = *(_DWORD *)(a1 + 1644);
            if ( (__int64 *)result != v43 )
            {
              v45 = 0;
              while ( a6 != *(_QWORD *)result )
              {
                if ( *(_QWORD *)result == -2 )
                  v45 = (_QWORD *)result;
                result += 8;
                if ( v43 == (__int64 *)result )
                {
                  if ( !v45 )
                    goto LABEL_87;
                  *v45 = a6;
                  --*(_DWORD *)(a1 + 1648);
                  ++*(_QWORD *)(a1 + 1616);
                  goto LABEL_40;
                }
              }
              goto LABEL_40;
            }
LABEL_87:
            if ( v44 < *(_DWORD *)(a1 + 1640) )
            {
              *(_DWORD *)(a1 + 1644) = v44 + 1;
              *v43 = a6;
              ++*(_QWORD *)(a1 + 1616);
            }
            else
            {
LABEL_46:
              v47 = v31;
              result = (__int64)sub_16CCBA0(v50, a6);
              v31 = v47;
            }
          }
LABEL_40:
          if ( *(_BYTE *)(v31 + 1) )
            result = sub_18DF510(a1, *(_QWORD *)(v31 + 24));
          goto LABEL_42;
        }
        result += 8LL * *(unsigned int *)(a1 + 1476);
        v36 = result;
      }
    }
    if ( v36 != result )
    {
      *(_QWORD *)result = -2;
      ++*(_DWORD *)(a1 + 1480);
    }
    goto LABEL_36;
  }
LABEL_3:
  if ( !*(_BYTE *)(result + 8) )
  {
    v15 = result;
    goto LABEL_14;
  }
  return result;
}
