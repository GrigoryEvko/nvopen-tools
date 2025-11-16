// Function: sub_2958540
// Address: 0x2958540
//
void __fastcall sub_2958540(char *src, char *a2, __int64 a3)
{
  char *i; // r13
  __int64 v7; // r12
  int v8; // ecx
  __int64 v9; // rdi
  __int64 v10; // r9
  unsigned int v11; // r11d
  int v12; // r8d
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // r10
  _QWORD **v16; // rax
  _QWORD *v17; // rdx
  unsigned int v18; // esi
  __int64 *v19; // rdx
  __int64 v20; // r10
  _QWORD **v21; // rdx
  _QWORD *v22; // rdx
  unsigned int j; // esi
  int v24; // edx
  char *v25; // r8
  __int64 v26; // rdi
  int v27; // ecx
  unsigned int v28; // edx
  __int64 *v29; // rax
  __int64 v30; // r10
  _QWORD **v31; // rax
  _QWORD *v32; // rdx
  unsigned int v33; // esi
  __int64 *v34; // rdx
  __int64 v35; // r10
  _QWORD **v36; // rdx
  _QWORD *v37; // rdx
  unsigned int k; // ecx
  int v39; // edx
  int v40; // eax
  int v41; // eax
  int v42; // esi
  int v43; // esi
  int v44; // [rsp-3Ch] [rbp-3Ch]
  int v45; // [rsp-3Ch] [rbp-3Ch]

  if ( src != a2 )
  {
    for ( i = src + 8; a2 != i; i += 8 )
    {
      while ( 1 )
      {
        v7 = *(_QWORD *)i;
        v8 = *(_DWORD *)(a3 + 24);
        v9 = *(_QWORD *)src;
        v10 = *(_QWORD *)(a3 + 8);
        v11 = ((unsigned int)*(_QWORD *)i >> 9) ^ ((unsigned int)*(_QWORD *)i >> 4);
        if ( !v8 )
          break;
        v12 = v8 - 1;
        v13 = v11 & (v8 - 1);
        v14 = (__int64 *)(v10 + 16LL * v13);
        v15 = *v14;
        if ( v7 == *v14 )
        {
LABEL_6:
          v16 = (_QWORD **)v14[1];
          if ( v16 )
          {
            v17 = *v16;
            for ( LODWORD(v16) = 1; v17; LODWORD(v16) = (_DWORD)v16 + 1 )
              v17 = (_QWORD *)*v17;
          }
        }
        else
        {
          v41 = 1;
          while ( v15 != -4096 )
          {
            v43 = v41 + 1;
            v13 = v12 & (v41 + v13);
            v14 = (__int64 *)(v10 + 16LL * v13);
            v15 = *v14;
            if ( v7 == *v14 )
              goto LABEL_6;
            v41 = v43;
          }
          LODWORD(v16) = 0;
        }
        v18 = v12 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v19 = (__int64 *)(v10 + 16LL * v18);
        v20 = *v19;
        if ( v9 != *v19 )
        {
          v24 = 1;
          while ( v20 != -4096 )
          {
            v18 = v12 & (v24 + v18);
            v45 = v24 + 1;
            v19 = (__int64 *)(v10 + 16LL * v18);
            v20 = *v19;
            if ( v9 == *v19 )
              goto LABEL_10;
            v24 = v45;
          }
          break;
        }
LABEL_10:
        v21 = (_QWORD **)v19[1];
        if ( !v21 )
          break;
        v22 = *v21;
        for ( j = 1; v22; ++j )
          v22 = (_QWORD *)*v22;
        if ( j <= (unsigned int)v16 )
          break;
        if ( src != i )
          memmove(src + 8, src, i - src);
        i += 8;
        *(_QWORD *)src = v7;
        if ( a2 == i )
          return;
      }
      v25 = i;
      v26 = *((_QWORD *)i - 1);
      if ( v8 )
      {
        while ( 1 )
        {
          v27 = v8 - 1;
          v28 = v27 & v11;
          v29 = (__int64 *)(v10 + 16LL * (v27 & v11));
          v30 = *v29;
          if ( v7 == *v29 )
          {
LABEL_22:
            v31 = (_QWORD **)v29[1];
            if ( v31 )
            {
              v32 = *v31;
              for ( LODWORD(v31) = 1; v32; LODWORD(v31) = (_DWORD)v31 + 1 )
                v32 = (_QWORD *)*v32;
            }
          }
          else
          {
            v40 = 1;
            while ( v30 != -4096 )
            {
              v42 = v40 + 1;
              v28 = v27 & (v40 + v28);
              v29 = (__int64 *)(v10 + 16LL * v28);
              v30 = *v29;
              if ( v7 == *v29 )
                goto LABEL_22;
              v40 = v42;
            }
            LODWORD(v31) = 0;
          }
          v33 = v27 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
          v34 = (__int64 *)(v10 + 16LL * v33);
          v35 = *v34;
          if ( v26 != *v34 )
            break;
LABEL_26:
          v36 = (_QWORD **)v34[1];
          if ( v36 )
          {
            v37 = *v36;
            for ( k = 1; v37; ++k )
              v37 = (_QWORD *)*v37;
            if ( k > (unsigned int)v31 )
            {
              *(_QWORD *)v25 = v26;
              v8 = *(_DWORD *)(a3 + 24);
              v25 -= 8;
              v10 = *(_QWORD *)(a3 + 8);
              v26 = *((_QWORD *)v25 - 1);
              if ( v8 )
                continue;
            }
          }
          goto LABEL_31;
        }
        v39 = 1;
        while ( v35 != -4096 )
        {
          v33 = v27 & (v39 + v33);
          v44 = v39 + 1;
          v34 = (__int64 *)(v10 + 16LL * v33);
          v35 = *v34;
          if ( v26 == *v34 )
            goto LABEL_26;
          v39 = v44;
        }
      }
LABEL_31:
      *(_QWORD *)v25 = v7;
    }
  }
}
