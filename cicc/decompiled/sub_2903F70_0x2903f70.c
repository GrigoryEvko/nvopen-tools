// Function: sub_2903F70
// Address: 0x2903f70
//
__int64 __fastcall sub_2903F70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 i, __int64 a6)
{
  __int64 v6; // r12
  __int64 result; // rax
  __int64 v9; // r15
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // r12
  __int64 v16; // rax
  unsigned __int64 v17; // r15
  _QWORD *v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  _QWORD *v22; // rbx
  _QWORD *v23; // r15
  __int64 v24; // rax
  unsigned __int64 v25; // rdi
  __int64 v26; // rax
  _QWORD *v27; // rax
  unsigned __int64 *v28; // r13
  unsigned __int64 v29; // rsi
  unsigned __int64 v30; // rsi
  _QWORD *v31; // r13
  _QWORD *j; // rbx
  __int64 v33; // [rsp+8h] [rbp-98h]
  __int64 v34; // [rsp+8h] [rbp-98h]
  unsigned int v35; // [rsp+14h] [rbp-8Ch]
  unsigned int v36; // [rsp+14h] [rbp-8Ch]
  unsigned int v37; // [rsp+20h] [rbp-80h]
  _QWORD *v38; // [rsp+20h] [rbp-80h]
  _QWORD *v39; // [rsp+20h] [rbp-80h]
  __int64 v40; // [rsp+28h] [rbp-78h]
  _QWORD v41[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v42; // [rsp+40h] [rbp-60h]
  _QWORD v43[2]; // [rsp+50h] [rbp-50h] BYREF
  __int64 v44; // [rsp+60h] [rbp-40h]

  v6 = *(_QWORD *)a1;
  result = *(_QWORD *)a1 + 112LL * *(unsigned int *)(a1 + 8);
  v40 = result;
  if ( *(_QWORD *)a1 != result )
  {
    do
    {
      while ( !a2 )
      {
        a2 = 112;
        v6 += 112;
        if ( v40 == v6 )
          goto LABEL_7;
      }
      v9 = a2 + 112;
      v10 = v6 + 112;
      sub_2900F20(a2, v6, a3, a4, i, a6);
      *(_QWORD *)(a2 + 48) = *(_QWORD *)(v6 + 48);
      v11 = *(_QWORD *)(v6 + 56);
      *(_DWORD *)(a2 + 88) = 0;
      *(_QWORD *)(a2 + 72) = 0;
      *(_DWORD *)(a2 + 80) = 0;
      *(_DWORD *)(a2 + 84) = 0;
      *(_QWORD *)(a2 + 56) = v11;
      *(_QWORD *)(a2 + 64) = 1;
      v12 = *(_QWORD *)(v6 + 72);
      ++*(_QWORD *)(v6 + 64);
      v13 = *(_QWORD *)(a2 + 72);
      *(_QWORD *)(a2 + 72) = v12;
      LODWORD(v12) = *(_DWORD *)(v6 + 80);
      *(_QWORD *)(v6 + 72) = v13;
      LODWORD(v13) = *(_DWORD *)(a2 + 80);
      *(_DWORD *)(a2 + 80) = v12;
      LODWORD(v12) = *(_DWORD *)(v6 + 84);
      *(_DWORD *)(v6 + 80) = v13;
      LODWORD(v13) = *(_DWORD *)(a2 + 84);
      *(_DWORD *)(a2 + 84) = v12;
      a3 = *(unsigned int *)(v6 + 88);
      *(_DWORD *)(v6 + 84) = v13;
      LODWORD(v13) = *(_DWORD *)(a2 + 88);
      *(_DWORD *)(a2 + 88) = a3;
      *(_DWORD *)(v6 + 88) = v13;
      *(_QWORD *)(a2 + 96) = a2 + 112;
      *(_DWORD *)(a2 + 104) = 0;
      *(_DWORD *)(a2 + 108) = 0;
      a6 = *(unsigned int *)(v6 + 104);
      if ( (_DWORD)a6 && a2 + 96 != v6 + 96 )
      {
        v26 = *(_QWORD *)(v6 + 96);
        if ( v10 == v26 )
        {
          v37 = *(_DWORD *)(v6 + 104);
          sub_2903DC0(a2 + 96, (unsigned int)a6, a3, a4, i, a6);
          v27 = *(_QWORD **)(v6 + 96);
          v28 = *(unsigned __int64 **)(a2 + 96);
          a6 = v37;
          for ( i = (__int64)&v27[6 * *(unsigned int *)(v6 + 104)]; (_QWORD *)i != v27; v28 += 6 )
          {
            if ( v28 )
            {
              *v28 = 0;
              v28[1] = 0;
              v29 = v27[2];
              v28[2] = v29;
              if ( v29 != 0 && v29 != -4096 && v29 != -8192 )
              {
                v38 = v27;
                v33 = i;
                v35 = a6;
                sub_BD6050(v28, *v27 & 0xFFFFFFFFFFFFFFF8LL);
                i = v33;
                a6 = v35;
                v27 = v38;
              }
              v28[3] = 0;
              v28[4] = 0;
              v30 = v27[5];
              v28[5] = v30;
              if ( v30 != 0 && v30 != -4096 && v30 != -8192 )
              {
                v39 = v27;
                v34 = i;
                v36 = a6;
                sub_BD6050(v28 + 3, v27[3] & 0xFFFFFFFFFFFFFFF8LL);
                i = v34;
                a6 = v36;
                v27 = v39;
              }
            }
            v27 += 6;
          }
          *(_DWORD *)(a2 + 104) = a6;
          v31 = *(_QWORD **)(v6 + 96);
          for ( j = &v31[6 * *(unsigned int *)(v6 + 104)]; v31 != j; sub_D68D70(j) )
          {
            j -= 6;
            sub_D68D70(j + 3);
          }
          *(_DWORD *)(v6 + 104) = 0;
        }
        else
        {
          *(_QWORD *)(a2 + 96) = v26;
          *(_DWORD *)(a2 + 104) = *(_DWORD *)(v6 + 104);
          *(_DWORD *)(a2 + 108) = *(_DWORD *)(v6 + 108);
          *(_QWORD *)(v6 + 96) = v10;
          *(_DWORD *)(v6 + 108) = 0;
          *(_DWORD *)(v6 + 104) = 0;
        }
      }
      a2 = v9;
      v6 += 112;
    }
    while ( v40 != v10 );
LABEL_7:
    v14 = *(_QWORD *)a1;
    result = *(unsigned int *)(a1 + 8);
    v15 = *(_QWORD *)a1 + 112 * result;
    if ( *(_QWORD *)a1 != v15 )
    {
      do
      {
        v16 = *(unsigned int *)(v15 - 8);
        v17 = *(_QWORD *)(v15 - 16);
        v15 -= 112;
        v18 = (_QWORD *)(v17 + 48 * v16);
        if ( (_QWORD *)v17 != v18 )
        {
          do
          {
            v19 = *(v18 - 1);
            v18 -= 6;
            if ( v19 != 0 && v19 != -4096 && v19 != -8192 )
              sub_BD60C0(v18 + 3);
            v20 = v18[2];
            if ( v20 != 0 && v20 != -4096 && v20 != -8192 )
              sub_BD60C0(v18);
          }
          while ( (_QWORD *)v17 != v18 );
          v17 = *(_QWORD *)(v15 + 96);
        }
        if ( v17 != v15 + 112 )
          _libc_free(v17);
        v21 = *(unsigned int *)(v15 + 88);
        if ( (_DWORD)v21 )
        {
          v41[0] = 0;
          v41[1] = 0;
          v42 = -4096;
          v43[0] = 0;
          v43[1] = 0;
          v44 = -8192;
          v22 = *(_QWORD **)(v15 + 72);
          v23 = &v22[4 * v21];
          do
          {
            v24 = v22[2];
            if ( v24 != -4096 && v24 != 0 && v24 != -8192 )
              sub_BD60C0(v22);
            v22 += 4;
          }
          while ( v23 != v22 );
          if ( v44 != 0 && v44 != -4096 && v44 != -8192 )
            sub_BD60C0(v43);
          if ( v42 != 0 && v42 != -4096 && v42 != -8192 )
            sub_BD60C0(v41);
        }
        sub_C7D6A0(*(_QWORD *)(v15 + 72), 32LL * *(unsigned int *)(v15 + 88), 8);
        v25 = *(_QWORD *)(v15 + 32);
        if ( v25 != v15 + 48 )
          _libc_free(v25);
        result = sub_C7D6A0(*(_QWORD *)(v15 + 8), 8LL * *(unsigned int *)(v15 + 24), 8);
      }
      while ( v15 != v14 );
    }
  }
  return result;
}
