// Function: sub_1A644B0
// Address: 0x1a644b0
//
__int64 __fastcall sub_1A644B0(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  __int64 result; // rax
  unsigned int v8; // esi
  __int64 v9; // rcx
  unsigned int v10; // edi
  unsigned int v11; // r8d
  __int64 *v12; // rdx
  __int64 v13; // r9
  __int64 v14; // r15
  unsigned int v15; // ebx
  _QWORD *v16; // rax
  unsigned int v17; // r9d
  __int64 *v18; // rdx
  __int64 v19; // r8
  int v20; // eax
  int v21; // edx
  int v22; // edx
  int v23; // edi
  __int64 v24; // r9
  unsigned int v25; // esi
  int v26; // ecx
  __int64 v27; // r8
  int v28; // r11d
  __int64 *v29; // r10
  int v30; // ecx
  int v31; // edx
  int v32; // esi
  __int64 v33; // r8
  __int64 *v34; // r9
  unsigned int v35; // ebx
  int v36; // r10d
  __int64 v37; // rdi
  int v38; // r10d
  int v39; // r11d
  __int64 *v40; // r10
  unsigned int v41; // [rsp-3Ch] [rbp-3Ch]
  unsigned int v42; // [rsp-3Ch] [rbp-3Ch]
  unsigned int v43; // [rsp-3Ch] [rbp-3Ch]

  result = (unsigned int)a3;
  if ( a3 <= dword_4FB48E0 )
  {
    v8 = *(_DWORD *)(a4 + 24);
    v9 = *(_QWORD *)(a4 + 8);
    if ( v8 )
    {
      v10 = v8 - 1;
      v11 = (v8 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v12 = (__int64 *)(v9 + 16LL * v11);
      v13 = *v12;
      if ( *v12 == a1 )
      {
LABEL_4:
        if ( v12 != (__int64 *)(v9 + 16LL * v8) )
          return *((unsigned int *)v12 + 2);
        v14 = *(_QWORD *)(a1 + 8);
        if ( !v14 )
        {
LABEL_14:
          v17 = v10 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
          v18 = (__int64 *)(v9 + 16LL * v17);
          v19 = *v18;
          if ( *v18 == a1 )
          {
LABEL_15:
            *((_DWORD *)v18 + 2) = result;
            return result;
          }
          v28 = 1;
          v29 = 0;
          while ( v19 != -8 )
          {
            if ( !v29 && v19 == -16 )
              v29 = v18;
            v17 = v10 & (v28 + v17);
            v18 = (__int64 *)(v9 + 16LL * v17);
            v19 = *v18;
            if ( *v18 == a1 )
              goto LABEL_15;
            ++v28;
          }
          v30 = *(_DWORD *)(a4 + 16);
          if ( v29 )
            v18 = v29;
          ++*(_QWORD *)a4;
          v26 = v30 + 1;
          if ( 4 * v26 < 3 * v8 )
          {
            if ( v8 - (v26 + *(_DWORD *)(a4 + 20)) > v8 >> 3 )
            {
LABEL_27:
              *(_DWORD *)(a4 + 16) = v26;
              if ( *v18 != -8 )
                --*(_DWORD *)(a4 + 20);
              *v18 = a1;
              *((_DWORD *)v18 + 2) = 0;
              goto LABEL_15;
            }
            v43 = result;
            sub_14672C0(a4, v8);
            v31 = *(_DWORD *)(a4 + 24);
            if ( v31 )
            {
              v32 = v31 - 1;
              v33 = *(_QWORD *)(a4 + 8);
              v34 = 0;
              v35 = (v31 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
              v36 = 1;
              v26 = *(_DWORD *)(a4 + 16) + 1;
              result = v43;
              v18 = (__int64 *)(v33 + 16LL * v35);
              v37 = *v18;
              if ( *v18 != a1 )
              {
                while ( v37 != -8 )
                {
                  if ( v37 == -16 && !v34 )
                    v34 = v18;
                  v35 = v32 & (v36 + v35);
                  v18 = (__int64 *)(v33 + 16LL * v35);
                  v37 = *v18;
                  if ( *v18 == a1 )
                    goto LABEL_27;
                  ++v36;
                }
                if ( v34 )
                  v18 = v34;
              }
              goto LABEL_27;
            }
LABEL_66:
            ++*(_DWORD *)(a4 + 16);
            BUG();
          }
LABEL_25:
          v42 = result;
          sub_14672C0(a4, 2 * v8);
          v22 = *(_DWORD *)(a4 + 24);
          if ( v22 )
          {
            v23 = v22 - 1;
            v24 = *(_QWORD *)(a4 + 8);
            v25 = (v22 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
            v26 = *(_DWORD *)(a4 + 16) + 1;
            result = v42;
            v18 = (__int64 *)(v24 + 16LL * v25);
            v27 = *v18;
            if ( *v18 != a1 )
            {
              v39 = 1;
              v40 = 0;
              while ( v27 != -8 )
              {
                if ( !v40 && v27 == -16 )
                  v40 = v18;
                v25 = v23 & (v39 + v25);
                v18 = (__int64 *)(v24 + 16LL * v25);
                v27 = *v18;
                if ( *v18 == a1 )
                  goto LABEL_27;
                ++v39;
              }
              if ( v40 )
                v18 = v40;
            }
            goto LABEL_27;
          }
          goto LABEL_66;
        }
      }
      else
      {
        v21 = 1;
        while ( v13 != -8 )
        {
          v38 = v21 + 1;
          v11 = v10 & (v21 + v11);
          v12 = (__int64 *)(v9 + 16LL * v11);
          v13 = *v12;
          if ( *v12 == a1 )
            goto LABEL_4;
          v21 = v38;
        }
        v14 = *(_QWORD *)(a1 + 8);
        if ( !v14 )
        {
          v15 = result;
          goto LABEL_13;
        }
      }
    }
    else
    {
      v14 = *(_QWORD *)(a1 + 8);
      if ( !v14 )
        goto LABEL_24;
    }
    v15 = result;
    v41 = result + 1;
    do
    {
      v16 = sub_1648700(v14);
      if ( *((_BYTE *)v16 + 16) > 0x17u && a2 == v16[5] )
      {
        v20 = sub_1A644B0(v16, a2, v41, a4);
        if ( (int)v15 < v20 )
          v15 = v20;
      }
      v14 = *(_QWORD *)(v14 + 8);
    }
    while ( v14 );
    v8 = *(_DWORD *)(a4 + 24);
    v9 = *(_QWORD *)(a4 + 8);
    if ( v8 )
    {
      v10 = v8 - 1;
LABEL_13:
      result = v15;
      goto LABEL_14;
    }
    LODWORD(result) = v15;
LABEL_24:
    ++*(_QWORD *)a4;
    v8 = 0;
    goto LABEL_25;
  }
  return result;
}
