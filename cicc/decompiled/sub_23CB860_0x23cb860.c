// Function: sub_23CB860
// Address: 0x23cb860
//
unsigned __int64 __fastcall sub_23CB860(_QWORD *a1, __int64 a2, int a3, int a4, int a5)
{
  __int64 v9; // rdx
  unsigned __int64 result; // rax
  __int64 v11; // rdx
  unsigned int v12; // esi
  __int64 v13; // r8
  _DWORD *v14; // r11
  int v15; // r15d
  unsigned int v16; // edi
  _DWORD *v17; // rdx
  int v18; // ecx
  unsigned __int64 *v19; // rdx
  int v20; // edx
  int v21; // edi
  __int64 v22; // r8
  unsigned int v23; // esi
  int v24; // ecx
  int v25; // edx
  int v26; // edi
  int v27; // edx
  int v28; // esi
  __int64 v29; // rdi
  _DWORD *v30; // r8
  unsigned int v31; // ebx
  int v32; // r9d
  int v33; // edx
  int v34; // r10d
  _DWORD *v35; // r9
  unsigned __int64 v36; // [rsp+8h] [rbp-38h]
  unsigned __int64 v37; // [rsp+8h] [rbp-38h]

  v9 = a1[3];
  a1[13] += 72LL;
  result = (v9 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[4] >= result + 72 && v9 )
    a1[3] = result + 72;
  else
    result = sub_9D1E70((__int64)(a1 + 3), 72, 72, 3);
  v11 = a1[27];
  *(_DWORD *)(result + 12) = a3;
  *(_DWORD *)(result + 8) = 1;
  *(_QWORD *)(result + 16) = 0xFFFFFFFF00000000LL;
  *(_DWORD *)(result + 24) = -1;
  *(_QWORD *)result = &unk_4A16248;
  *(_DWORD *)(result + 28) = a4;
  *(_QWORD *)(result + 32) = v11;
  *(_QWORD *)(result + 40) = 0;
  *(_QWORD *)(result + 48) = 0;
  *(_QWORD *)(result + 56) = 0;
  *(_DWORD *)(result + 64) = 0;
  if ( a2 )
  {
    v12 = *(_DWORD *)(a2 + 64);
    if ( v12 )
    {
      v13 = *(_QWORD *)(a2 + 48);
      v14 = 0;
      v15 = 1;
      v16 = (v12 - 1) & (37 * a5);
      v17 = (_DWORD *)(v13 + 16LL * v16);
      v18 = *v17;
      if ( *v17 == a5 )
      {
LABEL_7:
        v19 = (unsigned __int64 *)(v17 + 2);
LABEL_8:
        *v19 = result;
        return result;
      }
      while ( v18 != -1 )
      {
        if ( !v14 && v18 == -2 )
          v14 = v17;
        v16 = (v12 - 1) & (v15 + v16);
        v17 = (_DWORD *)(v13 + 16LL * v16);
        v18 = *v17;
        if ( *v17 == a5 )
          goto LABEL_7;
        ++v15;
      }
      v26 = *(_DWORD *)(a2 + 56);
      if ( !v14 )
        v14 = v17;
      ++*(_QWORD *)(a2 + 40);
      v24 = v26 + 1;
      if ( 4 * (v26 + 1) < 3 * v12 )
      {
        if ( v12 - *(_DWORD *)(a2 + 60) - v24 > v12 >> 3 )
        {
LABEL_14:
          *(_DWORD *)(a2 + 56) = v24;
          if ( *v14 != -1 )
            --*(_DWORD *)(a2 + 60);
          *v14 = a5;
          v19 = (unsigned __int64 *)(v14 + 2);
          *((_QWORD *)v14 + 1) = 0;
          goto LABEL_8;
        }
        v37 = result;
        sub_23CAE70(a2 + 40, v12);
        v27 = *(_DWORD *)(a2 + 64);
        if ( v27 )
        {
          v28 = v27 - 1;
          v29 = *(_QWORD *)(a2 + 48);
          v30 = 0;
          v31 = (v27 - 1) & (37 * a5);
          v32 = 1;
          v24 = *(_DWORD *)(a2 + 56) + 1;
          result = v37;
          v14 = (_DWORD *)(v29 + 16LL * v31);
          v33 = *v14;
          if ( *v14 != a5 )
          {
            while ( v33 != -1 )
            {
              if ( !v30 && v33 == -2 )
                v30 = v14;
              v31 = v28 & (v32 + v31);
              v14 = (_DWORD *)(v29 + 16LL * v31);
              v33 = *v14;
              if ( *v14 == a5 )
                goto LABEL_14;
              ++v32;
            }
            if ( v30 )
              v14 = v30;
          }
          goto LABEL_14;
        }
LABEL_48:
        ++*(_DWORD *)(a2 + 56);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a2 + 40);
    }
    v36 = result;
    sub_23CAE70(a2 + 40, 2 * v12);
    v20 = *(_DWORD *)(a2 + 64);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a2 + 48);
      v23 = (v20 - 1) & (37 * a5);
      v24 = *(_DWORD *)(a2 + 56) + 1;
      result = v36;
      v14 = (_DWORD *)(v22 + 16LL * v23);
      v25 = *v14;
      if ( *v14 != a5 )
      {
        v34 = 1;
        v35 = 0;
        while ( v25 != -1 )
        {
          if ( v25 == -2 && !v35 )
            v35 = v14;
          v23 = v21 & (v34 + v23);
          v14 = (_DWORD *)(v22 + 16LL * v23);
          v25 = *v14;
          if ( *v14 == a5 )
            goto LABEL_14;
          ++v34;
        }
        if ( v35 )
          v14 = v35;
      }
      goto LABEL_14;
    }
    goto LABEL_48;
  }
  return result;
}
