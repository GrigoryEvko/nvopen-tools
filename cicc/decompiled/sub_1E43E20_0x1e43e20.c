// Function: sub_1E43E20
// Address: 0x1e43e20
//
__int64 __fastcall sub_1E43E20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r12
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rsi
  char v13; // al
  unsigned int v14; // edx
  unsigned int v15; // eax
  bool v16; // cl
  unsigned int v17; // eax
  unsigned int v18; // edx
  int v19; // ecx
  int v20; // edx
  bool v21; // al
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rsi
  char v26; // al
  __int64 v27; // r14
  unsigned __int64 v28; // r13
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rdi
  __int64 v32; // rsi
  char v33; // dl
  __int64 v34; // r15
  unsigned __int64 v35; // r13
  __int64 v36; // r12
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdi
  __int64 v40; // rsi
  char v41; // al
  __int64 v43; // [rsp+0h] [rbp-40h]

  v7 = a1;
  v8 = a3;
  if ( a2 != a1 && a4 != a3 )
  {
    do
    {
      v14 = *(_DWORD *)(v8 + 60);
      v15 = *(_DWORD *)(v7 + 60);
      v16 = v14 > v15;
      if ( v14 != v15
        || (v17 = *(_DWORD *)(v8 + 72)) != 0 && (v18 = *(_DWORD *)(v7 + 72), v17 != v18) && (v16 = v17 < v18, v18) )
      {
        if ( !v16 )
          goto LABEL_16;
      }
      else
      {
        v19 = *(_DWORD *)(v8 + 64);
        v20 = *(_DWORD *)(v7 + 64);
        v21 = v19 < v20;
        if ( v19 == v20 )
          v21 = *(_DWORD *)(v8 + 68) > *(_DWORD *)(v7 + 68);
        if ( !v21 )
        {
LABEL_16:
          j___libc_free_0(*(_QWORD *)(a5 + 8));
          *(_DWORD *)(a5 + 24) = 0;
          *(_QWORD *)(a5 + 8) = 0;
          *(_DWORD *)(a5 + 16) = 0;
          *(_DWORD *)(a5 + 20) = 0;
          ++*(_QWORD *)a5;
          v22 = *(_QWORD *)(v7 + 8);
          ++*(_QWORD *)v7;
          v23 = *(_QWORD *)(a5 + 8);
          *(_QWORD *)(a5 + 8) = v22;
          LODWORD(v22) = *(_DWORD *)(v7 + 16);
          *(_QWORD *)(v7 + 8) = v23;
          LODWORD(v23) = *(_DWORD *)(a5 + 16);
          *(_DWORD *)(a5 + 16) = v22;
          LODWORD(v22) = *(_DWORD *)(v7 + 20);
          *(_DWORD *)(v7 + 16) = v23;
          LODWORD(v23) = *(_DWORD *)(a5 + 20);
          *(_DWORD *)(a5 + 20) = v22;
          LODWORD(v22) = *(_DWORD *)(v7 + 24);
          *(_DWORD *)(v7 + 20) = v23;
          LODWORD(v23) = *(_DWORD *)(a5 + 24);
          *(_DWORD *)(a5 + 24) = v22;
          *(_DWORD *)(v7 + 24) = v23;
          v24 = *(_QWORD *)(a5 + 32);
          v25 = *(_QWORD *)(a5 + 48);
          *(_QWORD *)(a5 + 32) = *(_QWORD *)(v7 + 32);
          *(_QWORD *)(a5 + 40) = *(_QWORD *)(v7 + 40);
          *(_QWORD *)(a5 + 48) = *(_QWORD *)(v7 + 48);
          *(_QWORD *)(v7 + 32) = 0;
          *(_QWORD *)(v7 + 40) = 0;
          *(_QWORD *)(v7 + 48) = 0;
          if ( v24 )
            j_j___libc_free_0(v24, v25 - v24);
          v26 = *(_BYTE *)(v7 + 56);
          v7 += 96;
          a5 += 96;
          *(_BYTE *)(a5 - 40) = v26;
          *(_DWORD *)(a5 - 36) = *(_DWORD *)(v7 - 36);
          *(_DWORD *)(a5 - 32) = *(_DWORD *)(v7 - 32);
          *(_DWORD *)(a5 - 28) = *(_DWORD *)(v7 - 28);
          *(_DWORD *)(a5 - 24) = *(_DWORD *)(v7 - 24);
          *(_QWORD *)(a5 - 16) = *(_QWORD *)(v7 - 16);
          *(_DWORD *)(a5 - 8) = *(_DWORD *)(v7 - 8);
          if ( a2 == v7 )
            break;
          continue;
        }
      }
      j___libc_free_0(*(_QWORD *)(a5 + 8));
      *(_DWORD *)(a5 + 24) = 0;
      *(_QWORD *)(a5 + 8) = 0;
      *(_DWORD *)(a5 + 16) = 0;
      *(_DWORD *)(a5 + 20) = 0;
      ++*(_QWORD *)a5;
      v9 = *(_QWORD *)(v8 + 8);
      ++*(_QWORD *)v8;
      v10 = *(_QWORD *)(a5 + 8);
      *(_QWORD *)(a5 + 8) = v9;
      LODWORD(v9) = *(_DWORD *)(v8 + 16);
      *(_QWORD *)(v8 + 8) = v10;
      LODWORD(v10) = *(_DWORD *)(a5 + 16);
      *(_DWORD *)(a5 + 16) = v9;
      LODWORD(v9) = *(_DWORD *)(v8 + 20);
      *(_DWORD *)(v8 + 16) = v10;
      LODWORD(v10) = *(_DWORD *)(a5 + 20);
      *(_DWORD *)(a5 + 20) = v9;
      LODWORD(v9) = *(_DWORD *)(v8 + 24);
      *(_DWORD *)(v8 + 20) = v10;
      LODWORD(v10) = *(_DWORD *)(a5 + 24);
      *(_DWORD *)(a5 + 24) = v9;
      *(_DWORD *)(v8 + 24) = v10;
      v11 = *(_QWORD *)(a5 + 32);
      v12 = *(_QWORD *)(a5 + 48);
      *(_QWORD *)(a5 + 32) = *(_QWORD *)(v8 + 32);
      *(_QWORD *)(a5 + 40) = *(_QWORD *)(v8 + 40);
      *(_QWORD *)(a5 + 48) = *(_QWORD *)(v8 + 48);
      *(_QWORD *)(v8 + 32) = 0;
      *(_QWORD *)(v8 + 40) = 0;
      *(_QWORD *)(v8 + 48) = 0;
      if ( v11 )
        j_j___libc_free_0(v11, v12 - v11);
      v13 = *(_BYTE *)(v8 + 56);
      a5 += 96;
      v8 += 96;
      *(_BYTE *)(a5 - 40) = v13;
      *(_DWORD *)(a5 - 36) = *(_DWORD *)(v8 - 36);
      *(_DWORD *)(a5 - 32) = *(_DWORD *)(v8 - 32);
      *(_DWORD *)(a5 - 28) = *(_DWORD *)(v8 - 28);
      *(_DWORD *)(a5 - 24) = *(_DWORD *)(v8 - 24);
      *(_QWORD *)(a5 - 16) = *(_QWORD *)(v8 - 16);
      *(_DWORD *)(a5 - 8) = *(_DWORD *)(v8 - 8);
      if ( a2 == v7 )
        break;
    }
    while ( a4 != v8 );
  }
  v43 = a2 - v7;
  v27 = a5;
  v28 = 0xAAAAAAAAAAAAAAABLL * (v43 >> 5);
  if ( v43 > 0 )
  {
    do
    {
      j___libc_free_0(*(_QWORD *)(v27 + 8));
      *(_DWORD *)(v27 + 24) = 0;
      *(_QWORD *)(v27 + 8) = 0;
      *(_DWORD *)(v27 + 16) = 0;
      *(_DWORD *)(v27 + 20) = 0;
      ++*(_QWORD *)v27;
      v29 = *(_QWORD *)(v7 + 8);
      ++*(_QWORD *)v7;
      v30 = *(_QWORD *)(v27 + 8);
      *(_QWORD *)(v27 + 8) = v29;
      LODWORD(v29) = *(_DWORD *)(v7 + 16);
      *(_QWORD *)(v7 + 8) = v30;
      LODWORD(v30) = *(_DWORD *)(v27 + 16);
      *(_DWORD *)(v27 + 16) = v29;
      LODWORD(v29) = *(_DWORD *)(v7 + 20);
      *(_DWORD *)(v7 + 16) = v30;
      LODWORD(v30) = *(_DWORD *)(v27 + 20);
      *(_DWORD *)(v27 + 20) = v29;
      LODWORD(v29) = *(_DWORD *)(v7 + 24);
      *(_DWORD *)(v7 + 20) = v30;
      LODWORD(v30) = *(_DWORD *)(v27 + 24);
      *(_DWORD *)(v27 + 24) = v29;
      *(_DWORD *)(v7 + 24) = v30;
      v31 = *(_QWORD *)(v27 + 32);
      v32 = *(_QWORD *)(v27 + 48);
      *(_QWORD *)(v27 + 32) = *(_QWORD *)(v7 + 32);
      *(_QWORD *)(v27 + 40) = *(_QWORD *)(v7 + 40);
      *(_QWORD *)(v27 + 48) = *(_QWORD *)(v7 + 48);
      *(_QWORD *)(v7 + 32) = 0;
      *(_QWORD *)(v7 + 40) = 0;
      *(_QWORD *)(v7 + 48) = 0;
      if ( v31 )
        j_j___libc_free_0(v31, v32 - v31);
      v33 = *(_BYTE *)(v7 + 56);
      v27 += 96;
      v7 += 96;
      *(_BYTE *)(v27 - 40) = v33;
      *(_DWORD *)(v27 - 36) = *(_DWORD *)(v7 - 36);
      *(_DWORD *)(v27 - 32) = *(_DWORD *)(v7 - 32);
      *(_DWORD *)(v27 - 28) = *(_DWORD *)(v7 - 28);
      *(_DWORD *)(v27 - 24) = *(_DWORD *)(v7 - 24);
      *(_QWORD *)(v27 - 16) = *(_QWORD *)(v7 - 16);
      *(_DWORD *)(v27 - 8) = *(_DWORD *)(v7 - 8);
      --v28;
    }
    while ( v28 );
    v27 = a5 + v43;
  }
  v34 = a4 - v8;
  v35 = 0xAAAAAAAAAAAAAAABLL * ((a4 - v8) >> 5);
  if ( a4 - v8 > 0 )
  {
    v36 = v27;
    do
    {
      j___libc_free_0(*(_QWORD *)(v36 + 8));
      ++*(_QWORD *)v36;
      *(_DWORD *)(v36 + 24) = 0;
      *(_QWORD *)(v36 + 8) = 0;
      *(_DWORD *)(v36 + 16) = 0;
      *(_DWORD *)(v36 + 20) = 0;
      v37 = *(_QWORD *)(v8 + 8);
      ++*(_QWORD *)v8;
      v38 = *(_QWORD *)(v36 + 8);
      *(_QWORD *)(v36 + 8) = v37;
      LODWORD(v37) = *(_DWORD *)(v8 + 16);
      *(_QWORD *)(v8 + 8) = v38;
      LODWORD(v38) = *(_DWORD *)(v36 + 16);
      *(_DWORD *)(v36 + 16) = v37;
      LODWORD(v37) = *(_DWORD *)(v8 + 20);
      *(_DWORD *)(v8 + 16) = v38;
      LODWORD(v38) = *(_DWORD *)(v36 + 20);
      *(_DWORD *)(v36 + 20) = v37;
      LODWORD(v37) = *(_DWORD *)(v8 + 24);
      *(_DWORD *)(v8 + 20) = v38;
      LODWORD(v38) = *(_DWORD *)(v36 + 24);
      *(_DWORD *)(v36 + 24) = v37;
      *(_DWORD *)(v8 + 24) = v38;
      v39 = *(_QWORD *)(v36 + 32);
      v40 = *(_QWORD *)(v36 + 48);
      *(_QWORD *)(v36 + 32) = *(_QWORD *)(v8 + 32);
      *(_QWORD *)(v36 + 40) = *(_QWORD *)(v8 + 40);
      *(_QWORD *)(v36 + 48) = *(_QWORD *)(v8 + 48);
      *(_QWORD *)(v8 + 32) = 0;
      *(_QWORD *)(v8 + 40) = 0;
      *(_QWORD *)(v8 + 48) = 0;
      if ( v39 )
        j_j___libc_free_0(v39, v40 - v39);
      v41 = *(_BYTE *)(v8 + 56);
      v36 += 96;
      v8 += 96;
      *(_BYTE *)(v36 - 40) = v41;
      *(_DWORD *)(v36 - 36) = *(_DWORD *)(v8 - 36);
      *(_DWORD *)(v36 - 32) = *(_DWORD *)(v8 - 32);
      *(_DWORD *)(v36 - 28) = *(_DWORD *)(v8 - 28);
      *(_DWORD *)(v36 - 24) = *(_DWORD *)(v8 - 24);
      *(_QWORD *)(v36 - 16) = *(_QWORD *)(v8 - 16);
      *(_DWORD *)(v36 - 8) = *(_DWORD *)(v8 - 8);
      --v35;
    }
    while ( v35 );
    if ( v34 <= 0 )
      v34 = 96;
    v27 += v34;
  }
  return v27;
}
