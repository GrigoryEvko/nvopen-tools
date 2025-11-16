// Function: sub_27207D0
// Address: 0x27207d0
//
__int64 __fastcall sub_27207D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r15
  __int64 v8; // r14
  unsigned int v9; // esi
  __int64 v10; // r9
  int v11; // r11d
  unsigned int v12; // edi
  __int64 v13; // rdx
  __int64 result; // rax
  __int64 v15; // rcx
  __int64 v16; // rbx
  int v17; // ecx
  int v18; // ecx
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r9
  __int64 v23; // rbx
  bool v24; // zf
  int v25; // eax
  int v26; // esi
  unsigned int v27; // edx
  __int64 v28; // rdi
  int v29; // r11d
  __int64 v30; // r10
  _QWORD *v31; // rdi
  __int64 v32; // rsi
  const void *v33; // r10
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // r14
  unsigned int v37; // r15d
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rsi
  __int64 v43; // rdi
  __int64 v44; // rsi
  __int64 v45; // rdi
  int v46; // eax
  int v47; // edx
  __int64 v48; // rdi
  unsigned int v49; // ebx
  int v50; // r10d
  __int64 v51; // rsi
  __int64 v52; // r10
  __int64 v53; // rdi
  int v54; // esi
  __int64 *v55; // r9
  __int64 v56; // r11
  __int64 v57; // rax
  _QWORD *v58; // rdi
  __int64 v59; // [rsp+0h] [rbp-70h]
  __int64 v60; // [rsp+8h] [rbp-68h]
  __int64 v61; // [rsp+18h] [rbp-58h]
  __int64 v62; // [rsp+18h] [rbp-58h]
  int v63; // [rsp+24h] [rbp-4Ch]
  const void *v64; // [rsp+28h] [rbp-48h]
  __int64 v65[7]; // [rsp+38h] [rbp-38h] BYREF

  v6 = a1 + 1608;
  v8 = a1 + 72;
  v9 = *(_DWORD *)(a1 + 96);
  v64 = (const void *)(a1 + 120);
  if ( !v9 )
    goto LABEL_27;
LABEL_2:
  v10 = *(_QWORD *)(a1 + 80);
  v11 = 1;
  v12 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = v10 + 24LL * v12;
  result = 0;
  v15 = *(_QWORD *)v13;
  if ( *(_QWORD *)v13 != a2 )
  {
    while ( v15 != -4096 )
    {
      if ( !result && v15 == -8192 )
        result = v13;
      a5 = (unsigned int)(v11 + 1);
      v12 = (v9 - 1) & (v11 + v12);
      v13 = v10 + 24LL * v12;
      v15 = *(_QWORD *)v13;
      if ( *(_QWORD *)v13 == a2 )
        goto LABEL_3;
      ++v11;
    }
    v17 = *(_DWORD *)(a1 + 88);
    if ( !result )
      result = v13;
    ++*(_QWORD *)(a1 + 72);
    v18 = v17 + 1;
    if ( 4 * v18 < 3 * v9 )
    {
      if ( v9 - *(_DWORD *)(a1 + 92) - v18 > v9 >> 3 )
        goto LABEL_15;
      sub_271E3C0(v8, v9);
      v46 = *(_DWORD *)(a1 + 96);
      if ( v46 )
      {
        v47 = v46 - 1;
        v48 = *(_QWORD *)(a1 + 80);
        v10 = 0;
        v49 = (v46 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v50 = 1;
        v18 = *(_DWORD *)(a1 + 88) + 1;
        result = v48 + 24LL * v49;
        v51 = *(_QWORD *)result;
        if ( *(_QWORD *)result != a2 )
        {
          while ( v51 != -4096 )
          {
            if ( !v10 && v51 == -8192 )
              v10 = result;
            a5 = (unsigned int)(v50 + 1);
            v49 = v47 & (v50 + v49);
            result = v48 + 24LL * v49;
            v51 = *(_QWORD *)result;
            if ( *(_QWORD *)result == a2 )
              goto LABEL_15;
            ++v50;
          }
          if ( v10 )
            result = v10;
        }
        goto LABEL_15;
      }
LABEL_91:
      ++*(_DWORD *)(a1 + 88);
      BUG();
    }
    while ( 1 )
    {
      sub_271E3C0(v8, 2 * v9);
      v25 = *(_DWORD *)(a1 + 96);
      if ( !v25 )
        goto LABEL_91;
      v26 = v25 - 1;
      v10 = *(_QWORD *)(a1 + 80);
      v27 = (v25 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v18 = *(_DWORD *)(a1 + 88) + 1;
      result = v10 + 24LL * v27;
      v28 = *(_QWORD *)result;
      if ( *(_QWORD *)result != a2 )
      {
        v29 = 1;
        v30 = 0;
        while ( v28 != -4096 )
        {
          if ( !v30 && v28 == -8192 )
            v30 = result;
          a5 = (unsigned int)(v29 + 1);
          v27 = v26 & (v29 + v27);
          result = v10 + 24LL * v27;
          v28 = *(_QWORD *)result;
          if ( *(_QWORD *)result == a2 )
            goto LABEL_15;
          ++v29;
        }
        if ( v30 )
          result = v30;
      }
LABEL_15:
      *(_DWORD *)(a1 + 88) = v18;
      if ( *(_QWORD *)result != -4096 )
        --*(_DWORD *)(a1 + 92);
      *(_QWORD *)result = a2;
      v16 = result + 8;
      *(_BYTE *)(result + 8) = 0;
      *(_QWORD *)(result + 16) = 0;
LABEL_18:
      *(_BYTE *)v16 = 1;
      v19 = *(unsigned int *)(a1 + 112);
      if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 116) )
      {
        sub_C8D5F0(a1 + 104, v64, v19 + 1, 8u, a5, v10);
        v19 = *(unsigned int *)(a1 + 112);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8 * v19) = a2;
      ++*(_DWORD *)(a1 + 112);
      result = sub_B10CD0(a2 + 48);
      if ( result )
        result = sub_271E5B0(a1, result, (__int64)v20, v21, a5, v22);
      v23 = *(_QWORD *)(v16 + 8);
      if ( *(_QWORD *)(v23 + 24) == a2 )
        break;
LABEL_23:
      if ( *(_BYTE *)v23 )
        return result;
      v24 = *(_BYTE *)(v23 + 3) == 0;
      *(_BYTE *)v23 = 1;
      if ( v24 )
      {
        *(_BYTE *)(v23 + 3) = 1;
        v44 = *(_QWORD *)(v23 + 16);
        if ( !*(_BYTE *)(a1 + 1636) )
          goto LABEL_63;
        result = *(_QWORD *)(a1 + 1616);
        v45 = *(unsigned int *)(a1 + 1628);
        v20 = (__int64 *)(result + 8 * v45);
        if ( (__int64 *)result == v20 )
        {
LABEL_61:
          if ( (unsigned int)v45 < *(_DWORD *)(a1 + 1624) )
          {
            *(_DWORD *)(a1 + 1628) = v45 + 1;
            *v20 = v44;
            ++*(_QWORD *)(a1 + 1608);
            goto LABEL_25;
          }
LABEL_63:
          result = (__int64)sub_C8CC70(v6, v44, (__int64)v20, v21, a5, v22);
          goto LABEL_25;
        }
        while ( v44 != *(_QWORD *)result )
        {
          result += 8;
          if ( v20 == (__int64 *)result )
            goto LABEL_61;
        }
      }
LABEL_25:
      if ( !*(_BYTE *)(v23 + 1) )
        return result;
      v9 = *(_DWORD *)(a1 + 96);
      a2 = *(_QWORD *)(v23 + 24);
      if ( v9 )
        goto LABEL_2;
LABEL_27:
      ++*(_QWORD *)(a1 + 72);
    }
    v20 = (__int64 *)(v23 + 16);
    if ( *(_DWORD *)(a1 + 1448) )
    {
      result = *(unsigned int *)(a1 + 1456);
      v52 = *(_QWORD *)(a1 + 1440);
      if ( !(_DWORD)result )
        goto LABEL_40;
      v53 = *(_QWORD *)(v23 + 16);
      v54 = result - 1;
      result = ((_DWORD)result - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
      v55 = (__int64 *)(v52 + 8 * result);
      v56 = *v55;
      if ( *v55 != v53 )
      {
        v22 = 1;
        while ( v56 != -4096 )
        {
          v21 = (unsigned int)(v22 + 1);
          result = v54 & (unsigned int)(result + v22);
          a5 = (unsigned int)result;
          v55 = (__int64 *)(v52 + 8LL * (unsigned int)result);
          v56 = *v55;
          if ( v53 == *v55 )
            goto LABEL_72;
          v22 = (unsigned int)v21;
        }
        goto LABEL_40;
      }
LABEL_72:
      *v55 = -8192;
      v57 = *(unsigned int *)(a1 + 1472);
      --*(_DWORD *)(a1 + 1448);
      v58 = *(_QWORD **)(a1 + 1464);
      ++*(_DWORD *)(a1 + 1452);
      v32 = (__int64)&v58[v57];
      result = (__int64)sub_271E080(v58, v32, v20);
      v33 = (const void *)(result + 8);
      if ( result + 8 == v32 )
        goto LABEL_39;
    }
    else
    {
      v31 = *(_QWORD **)(a1 + 1464);
      v32 = (__int64)&v31[*(unsigned int *)(a1 + 1472)];
      result = (__int64)sub_271E080(v31, v32, v20);
      if ( v32 == result )
      {
LABEL_40:
        if ( *(_BYTE *)(v23 + 1) )
          goto LABEL_23;
        v34 = *(_QWORD *)(a2 + 40);
        v35 = *(_QWORD *)(v34 + 48);
        v20 = (__int64 *)(v34 + 48);
        result = v35 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (__int64 *)result == v20 )
          goto LABEL_23;
        if ( !result )
          BUG();
        v21 = result - 24;
        result = (unsigned int)*(unsigned __int8 *)(result - 24) - 30;
        if ( (unsigned int)result > 0xA )
          goto LABEL_23;
        v61 = v21;
        result = sub_B46E30(v21);
        v63 = result;
        if ( !(_DWORD)result )
          goto LABEL_23;
        v60 = v8;
        v36 = v61;
        v62 = v6;
        v37 = 0;
        while ( 1 )
        {
          v65[0] = sub_B46EC0(v36, v37);
          result = sub_27204A0(a1 + 24, v65, v38, v39, v40, v41);
          v21 = result;
          if ( !*(_BYTE *)result )
            break;
LABEL_48:
          if ( ++v37 == v63 )
          {
            v8 = v60;
            v6 = v62;
            goto LABEL_23;
          }
        }
        v24 = *(_BYTE *)(result + 3) == 0;
        *(_BYTE *)result = 1;
        if ( v24 )
        {
          *(_BYTE *)(result + 3) = 1;
          v42 = *(_QWORD *)(result + 16);
          if ( !*(_BYTE *)(a1 + 1636) )
            goto LABEL_75;
          result = *(_QWORD *)(a1 + 1616);
          v43 = *(unsigned int *)(a1 + 1628);
          v20 = (__int64 *)(result + 8 * v43);
          if ( (__int64 *)result != v20 )
          {
            while ( v42 != *(_QWORD *)result )
            {
              result += 8;
              if ( v20 == (__int64 *)result )
                goto LABEL_55;
            }
            goto LABEL_46;
          }
LABEL_55:
          if ( (unsigned int)v43 < *(_DWORD *)(a1 + 1624) )
          {
            *(_DWORD *)(a1 + 1628) = v43 + 1;
            *v20 = v42;
            ++*(_QWORD *)(a1 + 1608);
          }
          else
          {
LABEL_75:
            v59 = v21;
            result = (__int64)sub_C8CC70(v62, v42, (__int64)v20, v21, a5, v22);
            v21 = v59;
          }
        }
LABEL_46:
        if ( *(_BYTE *)(v21 + 1) )
          result = sub_27207D0(a1, *(_QWORD *)(v21 + 24));
        goto LABEL_48;
      }
      v33 = (const void *)(result + 8);
      if ( v32 == result + 8 )
      {
LABEL_39:
        v22 = (unsigned int)(v22 - 1);
        *(_DWORD *)(a1 + 1472) = v22;
        goto LABEL_40;
      }
    }
    result = (__int64)memmove((void *)result, v33, v32 - (_QWORD)v33);
    LODWORD(v22) = *(_DWORD *)(a1 + 1472);
    goto LABEL_39;
  }
LABEL_3:
  v16 = v13 + 8;
  if ( !*(_BYTE *)(v13 + 8) )
    goto LABEL_18;
  return result;
}
