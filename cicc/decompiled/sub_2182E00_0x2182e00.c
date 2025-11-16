// Function: sub_2182E00
// Address: 0x2182e00
//
__int64 __fastcall sub_2182E00(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rdx
  __int64 result; // rax
  __int64 v9; // rbx
  __int64 v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rdi
  unsigned int v13; // ecx
  __int64 *v14; // rsi
  __int64 v15; // r8
  unsigned int v16; // r12d
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rdi
  unsigned int v20; // ecx
  __int64 *v21; // rsi
  __int64 v22; // r9
  unsigned int v23; // esi
  __int64 v24; // rdi
  unsigned int v25; // r10d
  __int64 *v26; // rdx
  __int64 v27; // rcx
  __int64 *v28; // r9
  int v29; // edi
  int v30; // ecx
  int v31; // esi
  int v32; // r10d
  __int64 v33; // rsi
  int i; // esi
  int v35; // r10d
  __int64 v36; // [rsp+18h] [rbp-78h]
  int v37; // [rsp+18h] [rbp-78h]
  __int64 v38; // [rsp+18h] [rbp-78h]
  __int64 v39; // [rsp+28h] [rbp-68h] BYREF
  _QWORD v40[12]; // [rsp+30h] [rbp-60h] BYREF

  v7 = *(_QWORD *)(a1 + 248);
  if ( a2 < 0 )
  {
    result = *(_QWORD *)(v7 + 24) + 16LL * (a2 & 0x7FFFFFFF);
    v9 = *(_QWORD *)(result + 8);
  }
  else
  {
    result = (unsigned int)a2;
    v9 = *(_QWORD *)(*(_QWORD *)(v7 + 272) + 8LL * (unsigned int)a2);
  }
  while ( v9 )
  {
    if ( (*(_BYTE *)(v9 + 3) & 0x10) == 0 && (*(_BYTE *)(v9 + 4) & 8) == 0 )
    {
      v10 = *(_QWORD *)(v9 + 16);
      result = **(unsigned __int16 **)(v10 + 16);
      if ( (_DWORD)result == 45 )
        goto LABEL_19;
LABEL_9:
      if ( (_DWORD)result )
      {
        result = *(_QWORD *)(v10 + 24);
        v39 = result;
        if ( a4 )
        {
          v11 = *(unsigned int *)(a4 + 24);
          if ( !(_DWORD)v11 )
            goto LABEL_15;
          v12 = *(_QWORD *)(a4 + 8);
          v13 = (v11 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
          v14 = (__int64 *)(v12 + 8LL * v13);
          v15 = *v14;
          if ( result != *v14 )
          {
            for ( i = 1; ; i = v35 )
            {
              if ( v15 == -8 )
                goto LABEL_15;
              v35 = i + 1;
              v13 = (v11 - 1) & (i + v13);
              v14 = (__int64 *)(v12 + 8LL * v13);
              v15 = *v14;
              if ( result == *v14 )
                break;
            }
          }
          result = v12 + 8 * v11;
          if ( v14 == (__int64 *)result )
            goto LABEL_15;
        }
        v36 = a4;
        result = sub_2182830((__int64)v40, a3, &v39);
        a4 = v36;
        goto LABEL_15;
      }
LABEL_19:
      while ( 1 )
      {
        v16 = 1;
        if ( *(_DWORD *)(v10 + 40) != 1 )
          break;
        do
        {
LABEL_15:
          v9 = *(_QWORD *)(v9 + 32);
          if ( !v9 )
            return result;
        }
        while ( (*(_BYTE *)(v9 + 3) & 0x10) != 0 || (*(_BYTE *)(v9 + 4) & 8) != 0 );
        v10 = *(_QWORD *)(v9 + 16);
        result = **(unsigned __int16 **)(v10 + 16);
        if ( (_DWORD)result != 45 )
          goto LABEL_9;
      }
      while ( 2 )
      {
        v17 = *(_QWORD *)(v10 + 32);
        result = v17 + 40LL * v16;
        if ( *(_BYTE *)result || a2 != *(_DWORD *)(result + 8) )
          goto LABEL_21;
        result = *(_QWORD *)(v17 + 40LL * (v16 + 1) + 24);
        v39 = result;
        if ( !a4 )
          goto LABEL_28;
        v18 = *(unsigned int *)(a4 + 24);
        if ( !(_DWORD)v18 )
          goto LABEL_21;
        v19 = *(_QWORD *)(a4 + 8);
        v20 = (v18 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
        v21 = (__int64 *)(v19 + 8LL * v20);
        v22 = *v21;
        if ( result != *v21 )
        {
          v31 = 1;
          while ( v22 != -8 )
          {
            v32 = v31 + 1;
            v33 = ((_DWORD)v18 - 1) & (v20 + v31);
            v20 = v33;
            v21 = (__int64 *)(v19 + 8 * v33);
            v22 = *v21;
            if ( result == *v21 )
              goto LABEL_27;
            v31 = v32;
          }
          goto LABEL_21;
        }
LABEL_27:
        if ( v21 == (__int64 *)(v19 + 8 * v18) )
          goto LABEL_21;
LABEL_28:
        v23 = *(_DWORD *)(a3 + 24);
        if ( v23 )
        {
          v24 = *(_QWORD *)(a3 + 8);
          v25 = (v23 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
          v26 = (__int64 *)(v24 + 8LL * v25);
          v27 = *v26;
          if ( result == *v26 )
            goto LABEL_21;
          v37 = 1;
          v28 = 0;
          while ( v27 != -8 )
          {
            if ( v27 != -16 || v28 )
              v26 = v28;
            v25 = (v23 - 1) & (v37 + v25);
            v27 = *(_QWORD *)(v24 + 8LL * v25);
            if ( result == v27 )
              goto LABEL_21;
            ++v37;
            v28 = v26;
            v26 = (__int64 *)(v24 + 8LL * v25);
          }
          v29 = *(_DWORD *)(a3 + 16);
          if ( !v28 )
            v28 = v26;
          ++*(_QWORD *)a3;
          v30 = v29 + 1;
          if ( 4 * (v29 + 1) < 3 * v23 )
          {
            if ( v23 - *(_DWORD *)(a3 + 20) - v30 > v23 >> 3 )
              goto LABEL_36;
            v38 = a4;
LABEL_46:
            sub_1DF9CE0(a3, v23);
            sub_1DF93E0(a3, &v39, v40);
            v28 = (__int64 *)v40[0];
            result = v39;
            a4 = v38;
            v30 = *(_DWORD *)(a3 + 16) + 1;
LABEL_36:
            *(_DWORD *)(a3 + 16) = v30;
            if ( *v28 != -8 )
              --*(_DWORD *)(a3 + 20);
            *v28 = result;
LABEL_21:
            v16 += 2;
            if ( *(_DWORD *)(v10 + 40) == v16 )
              goto LABEL_15;
            continue;
          }
        }
        else
        {
          ++*(_QWORD *)a3;
        }
        break;
      }
      v38 = a4;
      v23 *= 2;
      goto LABEL_46;
    }
    v9 = *(_QWORD *)(v9 + 32);
  }
  return result;
}
