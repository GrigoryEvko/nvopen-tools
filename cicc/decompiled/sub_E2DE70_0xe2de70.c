// Function: sub_E2DE70
// Address: 0xe2de70
//
unsigned __int64 __fastcall sub_E2DE70(__int64 a1, __int64 a2, unsigned int a3)
{
  char v6; // al
  __int64 v7; // rdi
  __int64 *v8; // rdi
  unsigned __int64 (__fastcall *v9)(__int64, char **, unsigned int); // rax
  unsigned __int64 result; // rax
  __int64 v11; // rdi
  char *v12; // r15
  __int64 v13; // r14
  __int64 v14; // rsi
  unsigned __int64 v15; // rax
  char *v16; // rdi
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  char *v20; // rdi
  __int64 v21; // rsi
  unsigned __int64 v22; // rdx
  char *v23; // rdi
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // rax
  char *v27; // rdi
  __int64 v28; // r14
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rax
  __int64 v31; // rax
  unsigned __int64 v32; // r8
  char *v33; // rdi
  unsigned int v34; // edx
  __int64 v35; // rsi
  __int64 v36; // rsi

  v6 = *(_BYTE *)(a1 + 24);
  switch ( v6 )
  {
    case 2:
      if ( (a3 & 4) == 0 )
      {
        v12 = "protected";
        v13 = 9;
        goto LABEL_23;
      }
      goto LABEL_14;
    case 3:
      if ( (a3 & 4) == 0 )
      {
        v12 = "public";
        v13 = 6;
        goto LABEL_23;
      }
LABEL_14:
      if ( (a3 & 8) != 0 )
        break;
      goto LABEL_15;
    case 1:
      v12 = "private";
      v13 = 7;
      if ( (a3 & 4) != 0 )
        goto LABEL_14;
LABEL_23:
      v21 = *(_QWORD *)(a2 + 8);
      v22 = *(_QWORD *)(a2 + 16);
      v23 = *(char **)a2;
      v24 = v13 + v21;
      if ( v13 + v21 > v22 )
      {
        v25 = 2 * v22;
        if ( v24 + 992 > v25 )
          *(_QWORD *)(a2 + 16) = v24 + 992;
        else
          *(_QWORD *)(a2 + 16) = v25;
        v26 = realloc(v23);
        *(_QWORD *)a2 = v26;
        v23 = (char *)v26;
        if ( !v26 )
          goto LABEL_49;
        v21 = *(_QWORD *)(a2 + 8);
      }
      v27 = &v23[v21];
      if ( (unsigned int)v13 >= 8 )
      {
        v32 = (unsigned __int64)(v27 + 8) & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)v27 = *(_QWORD *)v12;
        *(_QWORD *)&v27[v13 - 8] = *(_QWORD *)&v12[v13 - 8];
        v33 = &v27[-v32];
        if ( (((_DWORD)v13 + (_DWORD)v33) & 0xFFFFFFF8) >= 8 )
        {
          v34 = 0;
          do
          {
            v35 = v34;
            v34 += 8;
            *(_QWORD *)(v32 + v35) = *(_QWORD *)(v12 - v33 + v35);
          }
          while ( v34 < (((_DWORD)v13 + (_DWORD)v33) & 0xFFFFFFF8) );
        }
        v36 = *(_QWORD *)(a2 + 8);
      }
      else
      {
        *(_DWORD *)v27 = *(_DWORD *)v12;
        *(_DWORD *)&v27[(unsigned int)v13 - 4] = *(_DWORD *)&v12[(unsigned int)v13 - 4];
        v36 = *(_QWORD *)(a2 + 8);
      }
      v28 = v36 + v13;
      v29 = *(_QWORD *)(a2 + 16);
      *(_QWORD *)(a2 + 8) = v28;
      if ( v28 + 2 <= v29 )
      {
        v31 = *(_QWORD *)a2;
      }
      else
      {
        v30 = 2 * v29;
        if ( v28 + 994 > v30 )
          *(_QWORD *)(a2 + 16) = v28 + 994;
        else
          *(_QWORD *)(a2 + 16) = v30;
        v31 = realloc(*(void **)a2);
        *(_QWORD *)a2 = v31;
        if ( !v31 )
          goto LABEL_49;
        v28 = *(_QWORD *)(a2 + 8);
      }
      *(_WORD *)(v31 + v28) = 8250;
      *(_QWORD *)(a2 + 8) += 2LL;
      if ( (a3 & 8) != 0 )
        break;
LABEL_15:
      v14 = *(_QWORD *)(a2 + 8);
      v15 = *(_QWORD *)(a2 + 16);
      v16 = *(char **)a2;
      if ( v14 + 7 <= v15 )
      {
LABEL_20:
        v20 = &v16[v14];
        *(_DWORD *)v20 = 1952543859;
        *((_WORD *)v20 + 2) = 25449;
        v20[6] = 32;
        *(_QWORD *)(a2 + 8) += 7LL;
        break;
      }
      v17 = v14 + 999;
      v18 = 2 * v15;
      if ( v17 > v18 )
        *(_QWORD *)(a2 + 16) = v17;
      else
        *(_QWORD *)(a2 + 16) = v18;
      v19 = realloc(v16);
      *(_QWORD *)a2 = v19;
      v16 = (char *)v19;
      if ( v19 )
      {
        v14 = *(_QWORD *)(a2 + 8);
        goto LABEL_20;
      }
LABEL_49:
      abort();
  }
  if ( (a3 & 0x20) == 0 )
  {
    v7 = *(_QWORD *)(a1 + 32);
    if ( v7 )
    {
      (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v7 + 24LL))(v7, a2, a3);
      sub_E2A040(a2);
    }
  }
  v8 = *(__int64 **)(a1 + 16);
  v9 = *(unsigned __int64 (__fastcall **)(__int64, char **, unsigned int))(*v8 + 16);
  if ( v9 == sub_E2CA10 )
    result = sub_E2C8E0(v8[2], (char **)a2, a3, 2u, "::");
  else
    result = v9((__int64)v8, (char **)a2, a3);
  if ( (a3 & 0x20) == 0 )
  {
    v11 = *(_QWORD *)(a1 + 32);
    if ( v11 )
      return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v11 + 32LL))(v11, a2, a3);
  }
  return result;
}
