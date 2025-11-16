// Function: sub_3368280
// Address: 0x3368280
//
__int64 __fastcall sub_3368280(__int64 **a1, unsigned __int8 *a2)
{
  unsigned __int8 *v3; // rax
  unsigned __int8 *v4; // rbx
  __int64 v6; // rax
  __int64 v7; // rcx
  int v8; // eax
  int v9; // eax
  unsigned int v10; // r13d
  unsigned int v11; // esi
  unsigned __int8 *v12; // rdx
  __int64 v13; // r12
  char v14; // cl
  __int64 v15; // rdx
  int v16; // esi
  unsigned int v17; // r8d
  __int64 v18; // rax
  unsigned __int8 *v19; // rdi
  unsigned int v20; // esi
  unsigned int v21; // edx
  int v22; // edi
  unsigned int v23; // r8d
  int v24; // edi
  int v25; // r10d
  __int64 v26; // r9
  __int64 v27; // rdi
  int v28; // ecx
  unsigned int v29; // r13d
  unsigned __int8 *v30; // rdx
  __int64 v31; // r8
  int v32; // edx
  unsigned int v33; // r13d
  unsigned __int8 *v34; // rsi
  int v35; // edi
  __int64 v36; // rcx
  int v37; // ecx
  int v38; // edx
  int v39; // r8d
  __int64 v40; // rsi

  if ( !a2 )
    return 0;
  v3 = sub_BD3990(a2, (__int64)a2);
  v4 = v3;
  if ( *v3 == 60 && sub_B4D040((__int64)v3) )
  {
    v6 = **a1;
    v7 = *(_QWORD *)(v6 + 256);
    v8 = *(_DWORD *)(v6 + 272);
    if ( v8 )
    {
      v9 = v8 - 1;
      v10 = ((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4);
      v11 = v9 & v10;
      v12 = *(unsigned __int8 **)(v7 + 16LL * (v9 & v10));
      if ( v4 == v12 )
      {
LABEL_8:
        v13 = (__int64)a1[1];
        v14 = *(_BYTE *)(v13 + 8) & 1;
        if ( v14 )
        {
          v15 = v13 + 16;
          v16 = 7;
        }
        else
        {
          v20 = *(_DWORD *)(v13 + 24);
          v15 = *(_QWORD *)(v13 + 16);
          if ( !v20 )
          {
            v21 = *(_DWORD *)(v13 + 8);
            ++*(_QWORD *)v13;
            v18 = 0;
            v22 = (v21 >> 1) + 1;
            goto LABEL_16;
          }
          v16 = v20 - 1;
        }
        v17 = v16 & v10;
        v18 = v15 + 16LL * (v16 & v10);
        v19 = *(unsigned __int8 **)v18;
        if ( v4 == *(unsigned __int8 **)v18 )
          return v18 + 8;
        v25 = 1;
        v26 = 0;
        while ( v19 != (unsigned __int8 *)-4096LL )
        {
          if ( !v26 && v19 == (unsigned __int8 *)-8192LL )
            v26 = v18;
          v17 = v16 & (v25 + v17);
          v18 = v15 + 16LL * v17;
          v19 = *(unsigned __int8 **)v18;
          if ( v4 == *(unsigned __int8 **)v18 )
            return v18 + 8;
          ++v25;
        }
        v21 = *(_DWORD *)(v13 + 8);
        if ( v26 )
          v18 = v26;
        ++*(_QWORD *)v13;
        v22 = (v21 >> 1) + 1;
        if ( v14 )
        {
          v23 = 24;
          v20 = 8;
LABEL_17:
          if ( 4 * v22 < v23 )
          {
            if ( v20 - *(_DWORD *)(v13 + 12) - v22 > v20 >> 3 )
            {
LABEL_19:
              *(_DWORD *)(v13 + 8) = (2 * (v21 >> 1) + 2) | v21 & 1;
              if ( *(_QWORD *)v18 != -4096 )
                --*(_DWORD *)(v13 + 12);
              *(_QWORD *)v18 = v4;
              *(_DWORD *)(v18 + 8) = 0;
              return v18 + 8;
            }
            sub_3367360(v13, v20);
            if ( (*(_BYTE *)(v13 + 8) & 1) != 0 )
            {
              v31 = v13 + 16;
              v32 = 7;
              goto LABEL_38;
            }
            v38 = *(_DWORD *)(v13 + 24);
            v31 = *(_QWORD *)(v13 + 16);
            if ( v38 )
            {
              v32 = v38 - 1;
LABEL_38:
              v33 = v32 & v10;
              v18 = v31 + 16LL * v33;
              v34 = *(unsigned __int8 **)v18;
              if ( v4 != *(unsigned __int8 **)v18 )
              {
                v35 = 1;
                v36 = 0;
                while ( v34 != (unsigned __int8 *)-4096LL )
                {
                  if ( !v36 && v34 == (unsigned __int8 *)-8192LL )
                    v36 = v18;
                  v33 = v32 & (v35 + v33);
                  v18 = v31 + 16LL * v33;
                  v34 = *(unsigned __int8 **)v18;
                  if ( v4 == *(unsigned __int8 **)v18 )
                    goto LABEL_35;
                  ++v35;
                }
                if ( v36 )
                  v18 = v36;
              }
LABEL_35:
              v21 = *(_DWORD *)(v13 + 8);
              goto LABEL_19;
            }
LABEL_67:
            *(_DWORD *)(v13 + 8) = (2 * (*(_DWORD *)(v13 + 8) >> 1) + 2) | *(_DWORD *)(v13 + 8) & 1;
            BUG();
          }
          sub_3367360(v13, 2 * v20);
          if ( (*(_BYTE *)(v13 + 8) & 1) != 0 )
          {
            v27 = v13 + 16;
            v28 = 7;
          }
          else
          {
            v37 = *(_DWORD *)(v13 + 24);
            v27 = *(_QWORD *)(v13 + 16);
            if ( !v37 )
              goto LABEL_67;
            v28 = v37 - 1;
          }
          v29 = v28 & v10;
          v18 = v27 + 16LL * v29;
          v30 = *(unsigned __int8 **)v18;
          if ( v4 != *(unsigned __int8 **)v18 )
          {
            v39 = 1;
            v40 = 0;
            while ( v30 != (unsigned __int8 *)-4096LL )
            {
              if ( !v40 && v30 == (unsigned __int8 *)-8192LL )
                v40 = v18;
              v29 = v28 & (v39 + v29);
              v18 = v27 + 16LL * v29;
              v30 = *(unsigned __int8 **)v18;
              if ( v4 == *(unsigned __int8 **)v18 )
                goto LABEL_35;
              ++v39;
            }
            if ( v40 )
            {
              v21 = *(_DWORD *)(v13 + 8);
              v18 = v40;
              goto LABEL_19;
            }
          }
          goto LABEL_35;
        }
        v20 = *(_DWORD *)(v13 + 24);
LABEL_16:
        v23 = 3 * v20;
        goto LABEL_17;
      }
      v24 = 1;
      while ( v12 != (unsigned __int8 *)-4096LL )
      {
        v11 = v9 & (v24 + v11);
        v12 = *(unsigned __int8 **)(v7 + 16LL * v11);
        if ( v4 == v12 )
          goto LABEL_8;
        ++v24;
      }
    }
  }
  return 0;
}
