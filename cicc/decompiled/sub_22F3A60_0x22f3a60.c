// Function: sub_22F3A60
// Address: 0x22f3a60
//
__int64 __fastcall sub_22F3A60(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 result; // rax
  __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // r14
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rdx
  int v23; // r11d
  _DWORD *v24; // r10
  __int64 v25; // r8
  unsigned int v26; // edi
  _DWORD *v27; // rax
  int v28; // edx
  unsigned int v29; // ecx
  unsigned int v30; // edx
  __int64 v31; // rdx
  unsigned int v32; // esi
  int v33; // r15d
  int v34; // eax
  int v35; // ecx
  __int64 v36; // r8
  unsigned int v37; // edx
  int v38; // edi
  int v39; // esi
  int v40; // r10d
  _DWORD *v41; // r9
  int v42; // edi
  int v43; // eax
  int v44; // edx
  int v45; // r10d
  __int64 v46; // r8
  unsigned int v47; // ecx
  int v48; // esi
  _QWORD v49[8]; // [rsp+10h] [rbp-40h] BYREF

  v7 = *(unsigned int *)(a1 + 16);
  if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 20) )
  {
    sub_C8D5F0(a1 + 8, (const void *)(a1 + 24), v7 + 1, 8u, a5, a6);
    v7 = *(unsigned int *)(a1 + 16);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8 * v7) = a2;
  ++*(_DWORD *)(a1 + 16);
  result = sub_22F59B0(a2[1], *(unsigned __int16 *)(*a2 + 58));
  v10 = result;
  if ( result )
  {
    v11 = v9;
    v12 = sub_22F59B0(v9, *(unsigned __int16 *)(result + 58));
    v14 = v12;
    if ( !v12 )
      goto LABEL_19;
    v11 = v13;
    v15 = sub_22F59B0(v13, *(unsigned __int16 *)(v12 + 58));
    v10 = v15;
    if ( !v15 )
    {
      v10 = v14;
      goto LABEL_19;
    }
    v11 = v16;
    result = sub_22F59B0(v16, *(unsigned __int16 *)(v15 + 58));
    v18 = result;
    if ( result )
    {
      v11 = v17;
      result = sub_22F59B0(v17, *(unsigned __int16 *)(result + 58));
      v10 = result;
      if ( result )
      {
        v11 = v19;
        result = sub_22F59B0(v19, *(unsigned __int16 *)(result + 58));
        if ( result )
        {
          v11 = v20;
          v10 = result;
          result = sub_22F59B0(v20, *(unsigned __int16 *)(result + 58));
          v49[0] = result;
          v49[1] = v21;
          if ( result )
          {
            result = sub_22F33D0(v49);
            v10 = result;
            v11 = v22;
          }
        }
      }
      else
      {
        v10 = v18;
      }
    }
  }
  else
  {
    v10 = *a2;
    v11 = a2[1];
  }
  if ( v10 )
  {
LABEL_19:
    while ( 1 )
    {
      v32 = *(_DWORD *)(a1 + 176);
      v33 = *(_DWORD *)(v10 + 40);
      if ( !v32 )
        break;
      v23 = 1;
      v24 = 0;
      v25 = *(_QWORD *)(a1 + 160);
      v26 = (v32 - 1) & (37 * v33);
      v27 = (_DWORD *)(v25 + 12LL * v26);
      v28 = *v27;
      if ( v33 != *v27 )
      {
        while ( v28 != -1 )
        {
          if ( !v24 && v28 == -2 )
            v24 = v27;
          v26 = (v32 - 1) & (v23 + v26);
          v27 = (_DWORD *)(v25 + 12LL * v26);
          v28 = *v27;
          if ( v33 == *v27 )
            goto LABEL_15;
          ++v23;
        }
        v42 = *(_DWORD *)(a1 + 168);
        if ( v24 )
          v27 = v24;
        ++*(_QWORD *)(a1 + 152);
        v38 = v42 + 1;
        if ( 4 * v38 < 3 * v32 )
        {
          if ( v32 - *(_DWORD *)(a1 + 172) - v38 <= v32 >> 3 )
          {
            sub_22F3880(a1 + 152, v32);
            v43 = *(_DWORD *)(a1 + 176);
            if ( !v43 )
            {
LABEL_57:
              ++*(_DWORD *)(a1 + 168);
              BUG();
            }
            v44 = v43 - 1;
            v45 = 1;
            v41 = 0;
            v46 = *(_QWORD *)(a1 + 160);
            v47 = (v43 - 1) & (37 * v33);
            v38 = *(_DWORD *)(a1 + 168) + 1;
            v27 = (_DWORD *)(v46 + 12LL * v47);
            v48 = *v27;
            if ( v33 != *v27 )
            {
              while ( v48 != -1 )
              {
                if ( v48 == -2 && !v41 )
                  v41 = v27;
                v47 = v44 & (v45 + v47);
                v27 = (_DWORD *)(v46 + 12LL * v47);
                v48 = *v27;
                if ( v33 == *v27 )
                  goto LABEL_38;
                ++v45;
              }
LABEL_25:
              if ( v41 )
                v27 = v41;
            }
          }
LABEL_38:
          *(_DWORD *)(a1 + 168) = v38;
          if ( *v27 != -1 )
            --*(_DWORD *)(a1 + 172);
          *v27 = v33;
          *(_QWORD *)(v27 + 1) = 0xFFFFFFFFLL;
          v29 = -1;
          goto LABEL_16;
        }
LABEL_21:
        sub_22F3880(a1 + 152, 2 * v32);
        v34 = *(_DWORD *)(a1 + 176);
        if ( !v34 )
          goto LABEL_57;
        v35 = v34 - 1;
        v36 = *(_QWORD *)(a1 + 160);
        v37 = (v34 - 1) & (37 * v33);
        v38 = *(_DWORD *)(a1 + 168) + 1;
        v27 = (_DWORD *)(v36 + 12LL * v37);
        v39 = *v27;
        if ( v33 != *v27 )
        {
          v40 = 1;
          v41 = 0;
          while ( v39 != -1 )
          {
            if ( !v41 && v39 == -2 )
              v41 = v27;
            v37 = v35 & (v40 + v37);
            v27 = (_DWORD *)(v36 + 12LL * v37);
            v39 = *v27;
            if ( v33 == *v27 )
              goto LABEL_38;
            ++v40;
          }
          goto LABEL_25;
        }
        goto LABEL_38;
      }
LABEL_15:
      v29 = v27[1];
LABEL_16:
      v30 = *(_DWORD *)(a1 + 16) - 1;
      if ( v30 > v29 )
        v30 = v29;
      v27[1] = v30;
      v27[2] = *(_DWORD *)(a1 + 16);
      result = sub_22F59B0(v11, *(unsigned __int16 *)(v10 + 56));
      v10 = result;
      v11 = v31;
      if ( !result )
        return result;
    }
    ++*(_QWORD *)(a1 + 152);
    goto LABEL_21;
  }
  return result;
}
