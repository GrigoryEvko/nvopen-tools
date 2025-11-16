// Function: sub_2EE9BA0
// Address: 0x2ee9ba0
//
void __fastcall sub_2EE9BA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // rsi
  __int64 v9; // rbx
  unsigned __int64 v10; // rdx
  int v11; // r14d
  __int64 v12; // r15
  __int64 v13; // r12
  __int64 v14; // rbx
  unsigned __int64 v15; // r15
  unsigned __int64 i; // r12
  int v17; // eax
  __int64 v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rbx
  unsigned __int64 v22; // rdi
  __int64 v23; // r15
  unsigned __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // r15
  __int64 v27; // r12
  __int64 v28; // rcx
  __int64 v29; // rsi
  __int64 v30; // rdi
  __int64 v31; // rcx
  __int64 v32; // rbx
  __int64 v33; // r15
  __int64 v34; // rcx
  __int64 v35; // rsi
  __int64 v36; // rdi
  __int64 v37; // rcx
  __int64 v38; // [rsp-48h] [rbp-48h]
  unsigned __int64 v39; // [rsp-48h] [rbp-48h]
  __int64 v40; // [rsp-40h] [rbp-40h]
  unsigned __int64 v41; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v8 = *(unsigned int *)(a2 + 8);
    v9 = *(_QWORD *)a1;
    v10 = *(unsigned int *)(a1 + 8);
    v11 = v8;
    v12 = *(_QWORD *)a1;
    if ( v8 <= v10 )
    {
      v20 = *(_QWORD *)a1;
      if ( v8 )
      {
        v25 = *(_QWORD *)a2;
        v26 = v9 + 40;
        v38 = 88 * v8;
        v27 = *(_QWORD *)a2 + 40LL;
        v40 = v9 + 40 + 88 * v8;
        do
        {
          v28 = *(_QWORD *)(v27 - 40);
          v29 = v27;
          v30 = v26;
          v27 += 88;
          v26 += 88;
          *(_QWORD *)(v26 - 128) = v28;
          *(_QWORD *)(v26 - 120) = *(_QWORD *)(v27 - 120);
          *(_DWORD *)(v26 - 112) = *(_DWORD *)(v27 - 112);
          *(_DWORD *)(v26 - 108) = *(_DWORD *)(v27 - 108);
          *(_DWORD *)(v26 - 104) = *(_DWORD *)(v27 - 104);
          *(_DWORD *)(v26 - 100) = *(_DWORD *)(v27 - 100);
          *(_BYTE *)(v26 - 96) = *(_BYTE *)(v27 - 96);
          *(_BYTE *)(v26 - 95) = *(_BYTE *)(v27 - 95);
          v31 = *(unsigned int *)(v27 - 92);
          *(_DWORD *)(v26 - 92) = v31;
          sub_2EE7490(v30, v29, v25, v31, a5, a6);
        }
        while ( v40 != v26 );
        v20 = *(_QWORD *)a1;
        v10 = *(unsigned int *)(a1 + 8);
        v12 = v9 + v38;
      }
      v21 = v20 + 88 * v10;
      while ( v12 != v21 )
      {
        v21 -= 88;
        v22 = *(_QWORD *)(v21 + 40);
        if ( v22 != v21 + 56 )
          _libc_free(v22);
      }
    }
    else
    {
      if ( v8 > *(unsigned int *)(a1 + 12) )
      {
        v23 = v9 + 88 * v10;
        while ( v23 != v9 )
        {
          while ( 1 )
          {
            v23 -= 88;
            v24 = *(_QWORD *)(v23 + 40);
            if ( v24 == v23 + 56 )
              break;
            _libc_free(v24);
            if ( v23 == v9 )
              goto LABEL_22;
          }
        }
LABEL_22:
        *(_DWORD *)(a1 + 8) = 0;
        sub_2EE9700(a1, v8, v10, a4, a5, a6);
        v8 = *(unsigned int *)(a2 + 8);
        v9 = *(_QWORD *)a1;
        v10 = 0;
      }
      else if ( *(_DWORD *)(a1 + 8) )
      {
        v32 = v9 + 40;
        v10 *= 88LL;
        v33 = *(_QWORD *)a2 + 40LL;
        v39 = v32 + v10;
        do
        {
          v34 = *(_QWORD *)(v33 - 40);
          v35 = v33;
          v36 = v32;
          v41 = v10;
          v32 += 88;
          v33 += 88;
          *(_QWORD *)(v32 - 128) = v34;
          *(_QWORD *)(v32 - 120) = *(_QWORD *)(v33 - 120);
          *(_DWORD *)(v32 - 112) = *(_DWORD *)(v33 - 112);
          *(_DWORD *)(v32 - 108) = *(_DWORD *)(v33 - 108);
          *(_DWORD *)(v32 - 104) = *(_DWORD *)(v33 - 104);
          *(_DWORD *)(v32 - 100) = *(_DWORD *)(v33 - 100);
          *(_BYTE *)(v32 - 96) = *(_BYTE *)(v33 - 96);
          *(_BYTE *)(v32 - 95) = *(_BYTE *)(v33 - 95);
          v37 = *(unsigned int *)(v33 - 92);
          *(_DWORD *)(v32 - 92) = v37;
          sub_2EE7490(v36, v35, v10, v37, a5, a6);
          v10 = v41;
        }
        while ( v32 != v39 );
        v8 = *(unsigned int *)(a2 + 8);
        v9 = *(_QWORD *)a1;
      }
      v13 = *(_QWORD *)a2;
      v14 = v10 + v9;
      v15 = v13 + 88 * v8;
      for ( i = v10 + v13; v15 != i; i += 88LL )
      {
        while ( 1 )
        {
          if ( v14 )
          {
            *(_QWORD *)v14 = *(_QWORD *)i;
            *(_QWORD *)(v14 + 8) = *(_QWORD *)(i + 8);
            *(_DWORD *)(v14 + 16) = *(_DWORD *)(i + 16);
            *(_DWORD *)(v14 + 20) = *(_DWORD *)(i + 20);
            *(_DWORD *)(v14 + 24) = *(_DWORD *)(i + 24);
            *(_DWORD *)(v14 + 28) = *(_DWORD *)(i + 28);
            *(_BYTE *)(v14 + 32) = *(_BYTE *)(i + 32);
            *(_BYTE *)(v14 + 33) = *(_BYTE *)(i + 33);
            v17 = *(_DWORD *)(i + 36);
            *(_DWORD *)(v14 + 48) = 0;
            *(_DWORD *)(v14 + 36) = v17;
            *(_QWORD *)(v14 + 40) = v14 + 56;
            *(_DWORD *)(v14 + 52) = 4;
            if ( *(_DWORD *)(i + 48) )
              break;
          }
          i += 88LL;
          v14 += 88;
          if ( v15 == i )
            goto LABEL_11;
        }
        v18 = i + 40;
        v19 = v14 + 40;
        v14 += 88;
        sub_2EE7490(v19, v18, v10, a4, a5, a6);
      }
    }
LABEL_11:
    *(_DWORD *)(a1 + 8) = v11;
  }
}
