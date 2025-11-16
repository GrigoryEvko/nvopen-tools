// Function: sub_1058220
// Address: 0x1058220
//
char __fastcall sub_1058220(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rax
  __int64 v7; // r15
  __int64 v11; // rbx
  int v12; // edx
  __int64 v13; // rax
  _QWORD *v14; // rdi
  _QWORD *v15; // rsi
  __int64 v16; // rcx
  __int64 *v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // rdx
  _QWORD *v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // rcx
  __int64 v23; // rdi
  _QWORD *v24; // rax
  _QWORD *v25; // rdi
  _QWORD *v26; // rdx
  int v27; // ebx
  __int64 v28; // rsi
  __int64 v29; // rdi
  unsigned int v30; // edx
  __int64 v31; // r10
  unsigned int v32; // r11d
  __int64 v34; // [rsp+0h] [rbp-50h]
  __int64 v35; // [rsp+8h] [rbp-48h]
  __int64 v36[7]; // [rsp+18h] [rbp-38h] BYREF

  LOBYTE(v6) = a1 + 16;
  v7 = *(_QWORD *)(a2 + 16);
  v35 = a1 + 16;
  while ( v7 )
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)(v7 + 24);
      v12 = *(_DWORD *)(a3 + 72);
      v13 = *(_QWORD *)(v11 + 40);
      v36[0] = v13;
      if ( !v12 )
      {
        v14 = *(_QWORD **)(a3 + 88);
        v15 = &v14[*(unsigned int *)(a3 + 96)];
        LOBYTE(v6) = v15 != sub_1055FB0(v14, (__int64)v15, v36);
        goto LABEL_6;
      }
      v28 = *(_QWORD *)(a3 + 64);
      v16 = *(unsigned int *)(a3 + 80);
      v29 = v28 + 8 * v16;
      if ( (_DWORD)v16 )
        break;
LABEL_7:
      if ( *(_BYTE *)(a1 + 1284) )
      {
        v17 = *(__int64 **)(a1 + 1264);
        v18 = &v17[*(unsigned int *)(a1 + 1276)];
        if ( v17 != v18 )
        {
          while ( v11 != *v17 )
          {
            if ( v18 == ++v17 )
              goto LABEL_18;
          }
LABEL_12:
          v19 = *(unsigned int *)(a1 + 8);
          LODWORD(v6) = v19;
          if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v19 )
            goto LABEL_19;
          goto LABEL_13;
        }
      }
      else if ( sub_C8CA60(a1 + 1256, v11) )
      {
        goto LABEL_12;
      }
LABEL_18:
      sub_1057F60(a1, (unsigned __int8 *)v11, v18, v16, a5, a6);
      v19 = *(unsigned int *)(a1 + 8);
      LODWORD(v6) = v19;
      if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v19 )
      {
LABEL_19:
        v21 = v35;
        v22 = sub_C8D7D0(a1, v35, 0, 0x18u, (unsigned __int64 *)v36, a6);
        v23 = 24LL * *(unsigned int *)(a1 + 8);
        v24 = (_QWORD *)(v23 + v22);
        if ( v23 + v22 )
        {
          *v24 = a3;
          v24[1] = v11;
          v24[2] = a2;
          v23 = 24LL * *(unsigned int *)(a1 + 8);
        }
        v6 = *(_QWORD **)a1;
        v25 = (_QWORD *)(*(_QWORD *)a1 + v23);
        if ( *(_QWORD **)a1 != v25 )
        {
          v26 = (_QWORD *)v22;
          do
          {
            if ( v26 )
            {
              *v26 = *v6;
              v26[1] = v6[1];
              v21 = v6[2];
              v26[2] = v21;
            }
            v6 += 3;
            v26 += 3;
          }
          while ( v25 != v6 );
          v25 = *(_QWORD **)a1;
        }
        v27 = v36[0];
        if ( (_QWORD *)v35 != v25 )
        {
          v34 = v22;
          LOBYTE(v6) = _libc_free(v25, v21);
          v22 = v34;
        }
        ++*(_DWORD *)(a1 + 8);
        *(_QWORD *)a1 = v22;
        *(_DWORD *)(a1 + 12) = v27;
        goto LABEL_3;
      }
LABEL_13:
      v20 = (_QWORD *)(*(_QWORD *)a1 + 24 * v19);
      if ( v20 )
      {
        *v20 = a3;
        v20[1] = v11;
        v20[2] = a2;
        LODWORD(v6) = *(_DWORD *)(a1 + 8);
      }
      LODWORD(v6) = (_DWORD)v6 + 1;
      *(_DWORD *)(a1 + 8) = (_DWORD)v6;
      v7 = *(_QWORD *)(v7 + 8);
      if ( !v7 )
        return (char)v6;
    }
    v16 = (unsigned int)(v16 - 1);
    v30 = v16 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
    a5 = v28 + 8LL * v30;
    v31 = *(_QWORD *)a5;
    if ( v13 != *(_QWORD *)a5 )
    {
      a5 = 1;
      while ( v31 != -4096 )
      {
        v32 = a5 + 1;
        v30 = v16 & (a5 + v30);
        a5 = v28 + 8LL * v30;
        v31 = *(_QWORD *)a5;
        if ( v13 == *(_QWORD *)a5 )
          goto LABEL_32;
        a5 = v32;
      }
      goto LABEL_7;
    }
LABEL_32:
    LOBYTE(v6) = a5 != v29;
LABEL_6:
    if ( !(_BYTE)v6 )
      goto LABEL_7;
LABEL_3:
    v7 = *(_QWORD *)(v7 + 8);
  }
  return (char)v6;
}
