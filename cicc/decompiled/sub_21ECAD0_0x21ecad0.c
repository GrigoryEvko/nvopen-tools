// Function: sub_21ECAD0
// Address: 0x21ecad0
//
__int64 __fastcall sub_21ECAD0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v7; // rbx
  __int64 v8; // r15
  __int64 v9; // r15
  __int64 v10; // r12
  unsigned int v11; // esi
  unsigned int v12; // r8d
  __int64 v13; // rdi
  int v14; // r10d
  unsigned int v15; // edx
  unsigned int v16; // r12d
  _DWORD *v17; // rcx
  int v18; // r9d
  int v19; // r9d
  unsigned int v20; // ecx
  __int64 v21; // rdx
  __int64 v22; // rax
  unsigned int v23; // esi
  __int64 v24; // rdx
  unsigned int v25; // edi
  unsigned int v26; // r10d
  _DWORD *v27; // rcx
  int v28; // r8d
  int v29; // r8d
  unsigned int v30; // ecx
  __int64 v31; // rdx
  unsigned int v32; // ecx
  _DWORD *v33; // rdi
  int v34; // r9d
  unsigned int v35; // ecx
  __int64 v36; // rdx
  __int64 v37; // rax
  _DWORD *v38; // r8
  int v39; // edi
  int v40; // ecx
  int v41; // r9d
  _DWORD *v42; // r9
  int v43; // edi
  int v44; // ecx
  int v45; // r12d
  _DWORD *v46; // r10
  int v47; // edi
  int v48; // ecx
  int *v49; // [rsp-60h] [rbp-60h]
  int v50; // [rsp-58h] [rbp-58h]
  __int64 v51; // [rsp-58h] [rbp-58h]
  unsigned int v52; // [rsp-58h] [rbp-58h]
  int v53; // [rsp-58h] [rbp-58h]
  __int64 v54; // [rsp-58h] [rbp-58h]
  __int64 v55; // [rsp-58h] [rbp-58h]
  __int64 v56; // [rsp-50h] [rbp-50h]
  int v57; // [rsp-44h] [rbp-44h] BYREF
  _QWORD v58[8]; // [rsp-40h] [rbp-40h] BYREF

  result = (unsigned int)**(unsigned __int16 **)(a2 + 16) - 12;
  if ( (unsigned __int16)(**(_WORD **)(a2 + 16) - 12) > 1u )
  {
    result = *(unsigned int *)(a2 + 40);
    if ( (_DWORD)result )
    {
      v7 = 0;
      v8 = 5LL * (unsigned int)(result - 1) + 5;
      result = a1 + 56;
      v56 = a1 + 56;
      v9 = 8 * v8;
      do
      {
        while ( 1 )
        {
          v10 = v7 + *(_QWORD *)(a2 + 32);
          if ( *(_BYTE *)v10 )
            goto LABEL_14;
          result = *(unsigned int *)(v10 + 8);
          if ( (int)result >= 0 )
            goto LABEL_14;
          if ( (*(_BYTE *)(v10 + 3) & 0x10) != 0 )
            break;
LABEL_7:
          result = **(unsigned __int16 **)(a2 + 16);
          if ( !**(_WORD **)(a2 + 16) )
            goto LABEL_14;
          if ( (_DWORD)result == 45 )
            goto LABEL_14;
          v11 = *(_DWORD *)(a1 + 80);
          if ( !v11 )
            goto LABEL_14;
          result = *(unsigned int *)(v10 + 8);
          v12 = v11 - 1;
          v13 = *(_QWORD *)(a1 + 64);
          v14 = 1;
          v15 = (v11 - 1) & (37 * result);
          v16 = v15;
          v17 = (_DWORD *)(v13 + 8LL * v15);
          v18 = *v17;
          if ( (_DWORD)result != *v17 )
          {
            while ( v18 != -1 )
            {
              v16 = v12 & (v14 + v16);
              v18 = *(_DWORD *)(v13 + 8LL * v16);
              if ( (_DWORD)result == v18 )
                goto LABEL_11;
              ++v14;
            }
            goto LABEL_14;
          }
LABEL_11:
          v57 = result;
          v19 = *v17;
          if ( (_DWORD)result != *v17 )
          {
            v45 = 1;
            v46 = 0;
            while ( v19 != -1 )
            {
              if ( !v46 && v19 == -2 )
                v46 = v17;
              v15 = v12 & (v45 + v15);
              v17 = (_DWORD *)(v13 + 8LL * v15);
              v19 = *v17;
              if ( (_DWORD)result == *v17 )
                goto LABEL_12;
              ++v45;
            }
            v47 = *(_DWORD *)(a1 + 72);
            if ( !v46 )
              v46 = v17;
            ++*(_QWORD *)(a1 + 56);
            v48 = v47 + 1;
            if ( 4 * (v47 + 1) >= 3 * v11 )
            {
              v54 = a3;
              v11 *= 2;
            }
            else
            {
              if ( v11 - *(_DWORD *)(a1 + 76) - v48 > v11 >> 3 )
              {
LABEL_66:
                *(_DWORD *)(a1 + 72) = v48;
                if ( *v46 != -1 )
                  --*(_DWORD *)(a1 + 76);
                *v46 = result;
                v21 = 1;
                v22 = 0;
                v46[1] = 0;
                goto LABEL_13;
              }
              v54 = a3;
            }
            sub_1BFDD60(v56, v11);
            sub_1BFD720(v56, &v57, v58);
            v46 = (_DWORD *)v58[0];
            LODWORD(result) = v57;
            a3 = v54;
            v48 = *(_DWORD *)(a1 + 72) + 1;
            goto LABEL_66;
          }
LABEL_12:
          v20 = v17[1];
          v21 = 1LL << v20;
          v22 = 8LL * (v20 >> 6);
LABEL_13:
          result = *(_QWORD *)(a3 + 24) + v22;
          *(_QWORD *)result |= v21;
LABEL_14:
          v7 += 40;
          if ( v9 == v7 )
            return result;
        }
        v23 = *(_DWORD *)(a1 + 80);
        v24 = *(_QWORD *)(a1 + 64);
        if ( **(_WORD **)(a2 + 16) && **(_WORD **)(a2 + 16) != 45 )
        {
          if ( !v23 )
            goto LABEL_14;
          v25 = v23 - 1;
          v26 = (v23 - 1) & (37 * result);
          v27 = (_DWORD *)(v24 + 8LL * v26);
          v28 = *v27;
          if ( (_DWORD)result != *v27 )
          {
            v52 = (v23 - 1) & (37 * result);
            v41 = 1;
            while ( v28 != -1 )
            {
              v52 = v25 & (v52 + v41);
              v28 = *(_DWORD *)(v24 + 8LL * v52);
              if ( (_DWORD)result == v28 )
                goto LABEL_20;
              ++v41;
            }
            goto LABEL_14;
          }
LABEL_20:
          v57 = *(_DWORD *)(v10 + 8);
          v29 = *v27;
          if ( (_DWORD)result == *v27 )
          {
LABEL_21:
            v30 = v27[1];
            v31 = 8LL * (v30 >> 6);
            result = ~(1LL << v30);
LABEL_22:
            *(_QWORD *)(*(_QWORD *)(a3 + 24) + v31) &= result;
            goto LABEL_28;
          }
          v53 = 1;
          v42 = 0;
          while ( v29 != -1 )
          {
            if ( v29 == -2 && !v42 )
              v42 = v27;
            v26 = v25 & (v53 + v26);
            v27 = (_DWORD *)(v24 + 8LL * v26);
            v29 = *v27;
            if ( (_DWORD)result == *v27 )
              goto LABEL_21;
            ++v53;
          }
          v43 = *(_DWORD *)(a1 + 72);
          if ( !v42 )
            v42 = v27;
          ++*(_QWORD *)(a1 + 56);
          v44 = v43 + 1;
          if ( 4 * (v43 + 1) >= 3 * v23 )
          {
            v55 = a3;
            v23 *= 2;
          }
          else
          {
            if ( v23 - *(_DWORD *)(a1 + 76) - v44 > v23 >> 3 )
            {
LABEL_57:
              *(_DWORD *)(a1 + 72) = v44;
              if ( *v42 != -1 )
                --*(_DWORD *)(a1 + 76);
              *v42 = result;
              v31 = 0;
              result = -2;
              v42[1] = 0;
              goto LABEL_22;
            }
            v55 = a3;
          }
          sub_1BFDD60(v56, v23);
          sub_1BFD720(v56, &v57, v58);
          v42 = (_DWORD *)v58[0];
          LODWORD(result) = v57;
          a3 = v55;
          v44 = *(_DWORD *)(a1 + 72) + 1;
          goto LABEL_57;
        }
        v57 = *(_DWORD *)(v10 + 8);
        if ( v23 )
        {
          v32 = (v23 - 1) & (37 * result);
          v33 = (_DWORD *)(v24 + 8LL * v32);
          v34 = *v33;
          if ( (_DWORD)result == *v33 )
          {
            v35 = v33[1];
LABEL_26:
            v36 = 1LL << v35;
            v37 = 8LL * (v35 >> 6);
            goto LABEL_27;
          }
          v50 = 1;
          v38 = 0;
          while ( v34 != -1 )
          {
            if ( v34 != -2 || v38 )
              v33 = v38;
            v32 = (v23 - 1) & (v50 + v32);
            v49 = (int *)(v24 + 8LL * v32);
            v34 = *v49;
            if ( (_DWORD)result == *v49 )
            {
              v35 = v49[1];
              goto LABEL_26;
            }
            ++v50;
            v38 = v33;
            v33 = (_DWORD *)(v24 + 8LL * v32);
          }
          if ( !v38 )
            v38 = v33;
          v39 = *(_DWORD *)(a1 + 72);
          ++*(_QWORD *)(a1 + 56);
          v40 = v39 + 1;
          if ( 4 * (v39 + 1) < 3 * v23 )
          {
            if ( v23 - *(_DWORD *)(a1 + 76) - v40 > v23 >> 3 )
              goto LABEL_37;
            v51 = a3;
            goto LABEL_42;
          }
        }
        else
        {
          ++*(_QWORD *)(a1 + 56);
        }
        v51 = a3;
        v23 *= 2;
LABEL_42:
        sub_1BFDD60(v56, v23);
        sub_1BFD720(v56, &v57, v58);
        v38 = (_DWORD *)v58[0];
        LODWORD(result) = v57;
        a3 = v51;
        v40 = *(_DWORD *)(a1 + 72) + 1;
LABEL_37:
        *(_DWORD *)(a1 + 72) = v40;
        if ( *v38 != -1 )
          --*(_DWORD *)(a1 + 76);
        *v38 = result;
        v36 = 1;
        v37 = 0;
        v38[1] = 0;
LABEL_27:
        result = *(_QWORD *)(a3 + 24) + v37;
        *(_QWORD *)result |= v36;
LABEL_28:
        if ( (*(_BYTE *)(v10 + 3) & 0x10) == 0 )
          goto LABEL_7;
        v7 += 40;
      }
      while ( v9 != v7 );
    }
  }
  return result;
}
