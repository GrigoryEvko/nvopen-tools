// Function: sub_354AB10
// Address: 0x354ab10
//
void __fastcall sub_354AB10(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rcx
  __int64 v6; // r14
  _QWORD *v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rcx
  _QWORD *v11; // r13
  __int64 v12; // rdx
  char *v13; // r14
  char **v14; // rbx
  int v15; // eax
  int v16; // esi
  __int64 v17; // rdx
  char *v18; // rbx
  char v19; // dl
  __int64 v20; // rdx
  __int64 v21; // r9
  char *v22; // rax
  __int64 v23; // rdx
  char *v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rsi
  int v27; // edx
  __int64 v28; // rsi
  int v29; // edx
  char v30; // dl
  __int64 v31; // rdx
  int v32; // eax
  unsigned __int64 v33; // rdi
  int v34; // edx
  __int64 v35; // rsi
  char *v36; // [rsp-50h] [rbp-50h]
  __int64 v37; // [rsp-48h] [rbp-48h]
  __int64 v38; // [rsp-48h] [rbp-48h]
  __int64 v39; // [rsp-48h] [rbp-48h]
  __int64 v40; // [rsp-40h] [rbp-40h]
  char **v41; // [rsp-40h] [rbp-40h]

  v3 = 0x1745D1745D1745DLL;
  *a1 = a3;
  a1[1] = 0;
  a1[2] = 0;
  if ( a3 <= 0x1745D1745D1745DLL )
    v3 = a3;
  if ( a3 > 0 )
  {
    while ( 1 )
    {
      v40 = v3;
      v6 = 11 * v3;
      v7 = (_QWORD *)sub_2207800(88 * v3);
      v10 = v40;
      v11 = v7;
      if ( v7 )
        break;
      v3 = v40 >> 1;
      if ( !(v40 >> 1) )
        return;
    }
    *v7 = 1;
    v12 = *(_QWORD *)(a2 + 8);
    v13 = (char *)&v7[v6];
    v14 = (char **)(v7 + 4);
    v15 = *(_DWORD *)(a2 + 24);
    v16 = *(_DWORD *)(a2 + 40);
    *(_QWORD *)(a2 + 8) = 0;
    v11[1] = v12;
    v17 = *(_QWORD *)(a2 + 16);
    *((_DWORD *)v11 + 6) = v15;
    v11[4] = v11 + 6;
    ++*(_QWORD *)a2;
    v11[2] = v17;
    *(_QWORD *)(a2 + 16) = 0;
    *(_DWORD *)(a2 + 24) = 0;
    v11[5] = 0;
    v41 = (char **)(a2 + 32);
    if ( v16 )
    {
      v39 = v10;
      sub_353DE10((__int64)(v11 + 4), v41, v17, v10, v8, v9);
      v10 = v39;
    }
    *((_BYTE *)v11 + 48) = *(_BYTE *)(a2 + 48);
    *(_QWORD *)((char *)v11 + 52) = *(_QWORD *)(a2 + 52);
    *(_QWORD *)((char *)v11 + 60) = *(_QWORD *)(a2 + 60);
    v11[9] = *(_QWORD *)(a2 + 72);
    *((_DWORD *)v11 + 20) = *(_DWORD *)(a2 + 80);
    if ( v13 == (char *)(v11 + 11) )
    {
      v22 = (char *)v11;
    }
    else
    {
      v18 = (char *)(v11 + 11);
      do
      {
        v20 = *((_QWORD *)v18 - 10);
        ++*((_QWORD *)v18 - 11);
        v21 = (__int64)(v18 - 88);
        v22 = v18;
        *(_QWORD *)v18 = 1;
        *((_QWORD *)v18 + 1) = v20;
        LODWORD(v20) = *((_DWORD *)v18 - 18);
        *((_QWORD *)v18 - 10) = 0;
        *((_DWORD *)v18 + 4) = v20;
        LODWORD(v20) = *((_DWORD *)v18 - 17);
        *((_DWORD *)v18 - 18) = 0;
        *((_DWORD *)v18 + 5) = v20;
        LODWORD(v20) = *((_DWORD *)v18 - 16);
        *((_DWORD *)v18 - 17) = 0;
        *((_DWORD *)v18 + 6) = v20;
        *((_QWORD *)v18 + 4) = v18 + 48;
        v23 = *((unsigned int *)v18 - 12);
        *((_DWORD *)v18 - 16) = 0;
        *((_DWORD *)v18 + 10) = 0;
        *((_DWORD *)v18 + 11) = 0;
        if ( (_DWORD)v23 )
        {
          v37 = v10;
          sub_353DE10((__int64)(v18 + 32), (char **)v18 - 7, v23, v10, v8, v21);
          v21 = (__int64)(v18 - 88);
          v22 = v18;
          v10 = v37;
        }
        v19 = *(v18 - 40);
        v18 += 88;
        *(v18 - 40) = v19;
        *((_DWORD *)v18 - 9) = *((_DWORD *)v18 - 31);
        *((_DWORD *)v18 - 8) = *((_DWORD *)v18 - 30);
        *((_DWORD *)v18 - 7) = *((_DWORD *)v18 - 29);
        *((_DWORD *)v18 - 6) = *((_DWORD *)v18 - 28);
        *((_QWORD *)v18 - 2) = *((_QWORD *)v18 - 13);
        *((_DWORD *)v18 - 2) = *((_DWORD *)v18 - 24);
      }
      while ( v13 != v18 );
      v14 = (char **)(v21 + 120);
    }
    v36 = v22;
    v38 = v10;
    sub_C7D6A0(*(_QWORD *)(a2 + 8), 8LL * *(unsigned int *)(a2 + 24), 8);
    v24 = v36;
    ++*(_QWORD *)a2;
    v25 = v38;
    v26 = *((_QWORD *)v36 + 1);
    v27 = *((_DWORD *)v36 + 6);
    *((_QWORD *)v36 + 1) = 0;
    ++*(_QWORD *)v36;
    *(_QWORD *)(a2 + 8) = v26;
    v28 = *((_QWORD *)v36 + 2);
    *(_DWORD *)(a2 + 24) = v27;
    *(_QWORD *)(a2 + 16) = v28;
    *((_QWORD *)v36 + 2) = 0;
    *((_DWORD *)v36 + 6) = 0;
    if ( v14 != v41 )
    {
      v29 = *((_DWORD *)v36 + 10);
      if ( v29 )
      {
        v33 = *(_QWORD *)(a2 + 32);
        if ( v33 != a2 + 48 )
        {
          _libc_free(v33);
          v24 = v36;
          v25 = v38;
          v29 = *((_DWORD *)v36 + 10);
        }
        *(_DWORD *)(a2 + 40) = v29;
        v34 = *((_DWORD *)v24 + 11);
        v35 = *((_QWORD *)v24 + 4);
        *((_QWORD *)v24 + 5) = 0;
        *(_DWORD *)(a2 + 44) = v34;
        *(_QWORD *)(a2 + 32) = v35;
        *((_QWORD *)v24 + 4) = v24 + 48;
      }
      else
      {
        *(_DWORD *)(a2 + 40) = 0;
      }
    }
    v30 = v24[48];
    a1[2] = (__int64)v11;
    a1[1] = v25;
    *(_BYTE *)(a2 + 48) = v30;
    *(_QWORD *)(a2 + 52) = *(_QWORD *)(v24 + 52);
    *(_QWORD *)(a2 + 60) = *(_QWORD *)(v24 + 60);
    v31 = *((_QWORD *)v24 + 9);
    v32 = *((_DWORD *)v24 + 20);
    *(_QWORD *)(a2 + 72) = v31;
    *(_DWORD *)(a2 + 80) = v32;
  }
}
