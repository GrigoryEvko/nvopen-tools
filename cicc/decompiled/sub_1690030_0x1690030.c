// Function: sub_1690030
// Address: 0x1690030
//
__int64 __fastcall sub_1690030(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // r14
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdx
  int v19; // r11d
  _DWORD *v20; // r10
  __int64 v21; // r8
  unsigned int v22; // edi
  _DWORD *v23; // rax
  int v24; // edx
  unsigned int v25; // ecx
  unsigned int v26; // edx
  __int64 v27; // rdx
  unsigned int v28; // esi
  int v29; // r15d
  int v30; // eax
  int v31; // ecx
  __int64 v32; // r8
  unsigned int v33; // edx
  int v34; // edi
  int v35; // esi
  int v36; // r10d
  _DWORD *v37; // r9
  int v38; // edi
  int v39; // eax
  int v40; // edx
  int v41; // r10d
  __int64 v42; // r8
  unsigned int v43; // ecx
  int v44; // esi
  _QWORD v45[8]; // [rsp+10h] [rbp-40h] BYREF

  v3 = *(unsigned int *)(a1 + 16);
  if ( (unsigned int)v3 >= *(_DWORD *)(a1 + 20) )
  {
    sub_16CD150(a1 + 8, a1 + 24, 0, 8);
    v3 = *(unsigned int *)(a1 + 16);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8 * v3) = a2;
  ++*(_DWORD *)(a1 + 16);
  result = sub_1691920(a2[1], *(unsigned __int16 *)(*a2 + 42));
  v6 = result;
  if ( result )
  {
    v7 = v5;
    v8 = sub_1691920(v5, *(unsigned __int16 *)(result + 42));
    v10 = v8;
    if ( !v8 )
      goto LABEL_19;
    v7 = v9;
    v11 = sub_1691920(v9, *(unsigned __int16 *)(v8 + 42));
    v6 = v11;
    if ( !v11 )
    {
      v6 = v10;
      goto LABEL_19;
    }
    v7 = v12;
    result = sub_1691920(v12, *(unsigned __int16 *)(v11 + 42));
    v14 = result;
    if ( result )
    {
      v7 = v13;
      result = sub_1691920(v13, *(unsigned __int16 *)(result + 42));
      v6 = result;
      if ( result )
      {
        v7 = v15;
        result = sub_1691920(v15, *(unsigned __int16 *)(result + 42));
        if ( result )
        {
          v7 = v16;
          v6 = result;
          result = sub_1691920(v16, *(unsigned __int16 *)(result + 42));
          v45[0] = result;
          v45[1] = v17;
          if ( result )
          {
            result = sub_168F990(v45);
            v6 = result;
            v7 = v18;
          }
        }
      }
      else
      {
        v6 = v14;
      }
    }
  }
  else
  {
    v6 = *a2;
    v7 = a2[1];
  }
  if ( v6 )
  {
LABEL_19:
    while ( 1 )
    {
      v28 = *(_DWORD *)(a1 + 176);
      v29 = *(_DWORD *)(v6 + 32);
      if ( !v28 )
        break;
      v19 = 1;
      v20 = 0;
      v21 = *(_QWORD *)(a1 + 160);
      v22 = (v28 - 1) & (37 * v29);
      v23 = (_DWORD *)(v21 + 12LL * v22);
      v24 = *v23;
      if ( v29 != *v23 )
      {
        while ( v24 != -1 )
        {
          if ( !v20 && v24 == -2 )
            v20 = v23;
          v22 = (v28 - 1) & (v19 + v22);
          v23 = (_DWORD *)(v21 + 12LL * v22);
          v24 = *v23;
          if ( v29 == *v23 )
            goto LABEL_15;
          ++v19;
        }
        v38 = *(_DWORD *)(a1 + 168);
        if ( v20 )
          v23 = v20;
        ++*(_QWORD *)(a1 + 152);
        v34 = v38 + 1;
        if ( 4 * v34 < 3 * v28 )
        {
          if ( v28 - *(_DWORD *)(a1 + 172) - v34 <= v28 >> 3 )
          {
            sub_168FE70(a1 + 152, v28);
            v39 = *(_DWORD *)(a1 + 176);
            if ( !v39 )
            {
LABEL_57:
              ++*(_DWORD *)(a1 + 168);
              BUG();
            }
            v40 = v39 - 1;
            v41 = 1;
            v37 = 0;
            v42 = *(_QWORD *)(a1 + 160);
            v43 = (v39 - 1) & (37 * v29);
            v34 = *(_DWORD *)(a1 + 168) + 1;
            v23 = (_DWORD *)(v42 + 12LL * v43);
            v44 = *v23;
            if ( v29 != *v23 )
            {
              while ( v44 != -1 )
              {
                if ( v44 == -2 && !v37 )
                  v37 = v23;
                v43 = v40 & (v41 + v43);
                v23 = (_DWORD *)(v42 + 12LL * v43);
                v44 = *v23;
                if ( v29 == *v23 )
                  goto LABEL_38;
                ++v41;
              }
LABEL_25:
              if ( v37 )
                v23 = v37;
            }
          }
LABEL_38:
          *(_DWORD *)(a1 + 168) = v34;
          if ( *v23 != -1 )
            --*(_DWORD *)(a1 + 172);
          *v23 = v29;
          *(_QWORD *)(v23 + 1) = 0xFFFFFFFFLL;
          v25 = -1;
          goto LABEL_16;
        }
LABEL_21:
        sub_168FE70(a1 + 152, 2 * v28);
        v30 = *(_DWORD *)(a1 + 176);
        if ( !v30 )
          goto LABEL_57;
        v31 = v30 - 1;
        v32 = *(_QWORD *)(a1 + 160);
        v33 = (v30 - 1) & (37 * v29);
        v34 = *(_DWORD *)(a1 + 168) + 1;
        v23 = (_DWORD *)(v32 + 12LL * v33);
        v35 = *v23;
        if ( v29 != *v23 )
        {
          v36 = 1;
          v37 = 0;
          while ( v35 != -1 )
          {
            if ( !v37 && v35 == -2 )
              v37 = v23;
            v33 = v31 & (v36 + v33);
            v23 = (_DWORD *)(v32 + 12LL * v33);
            v35 = *v23;
            if ( v29 == *v23 )
              goto LABEL_38;
            ++v36;
          }
          goto LABEL_25;
        }
        goto LABEL_38;
      }
LABEL_15:
      v25 = v23[1];
LABEL_16:
      v26 = *(_DWORD *)(a1 + 16) - 1;
      if ( v26 > v25 )
        v26 = v25;
      v23[1] = v26;
      v23[2] = *(_DWORD *)(a1 + 16);
      result = sub_1691920(v7, *(unsigned __int16 *)(v6 + 40));
      v6 = result;
      v7 = v27;
      if ( !result )
        return result;
    }
    ++*(_QWORD *)(a1 + 152);
    goto LABEL_21;
  }
  return result;
}
