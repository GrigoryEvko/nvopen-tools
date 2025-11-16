// Function: sub_2E33150
// Address: 0x2e33150
//
__int64 __fastcall sub_2E33150(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r8
  __int64 *v7; // rbx
  __int64 v8; // rdi
  __int64 (*v9)(); // rax
  __int64 v10; // rdi
  __int16 v11; // bx
  unsigned __int16 v12; // ax
  int v13; // r12d
  int v14; // r14d
  __int64 *v15; // rbx
  __int64 v16; // rax
  __int64 *v17; // r15
  _DWORD *v18; // rax
  __int64 v19; // rdx
  __int64 *v21; // rax
  __int64 v22; // rdi
  __int64 *v23; // rdx
  __int64 v24; // rax
  __int64 v25; // r8
  __int64 v26; // rsi
  __int64 v27; // rdx
  __int64 (*v28)(); // rax
  __int64 (*v29)(); // rdx
  __int64 *v30; // rbx
  __int64 v31; // rdx
  _DWORD *v32; // rax
  __int64 v33; // rdi
  __int64 *v34; // rdx
  unsigned __int16 v35; // ax
  __int64 v36; // [rsp+8h] [rbp-38h]
  __int64 *v37; // [rsp+8h] [rbp-38h]

  v4 = 0;
  v7 = *(__int64 **)(a2 + 32);
  v8 = v7[2];
  v9 = *(__int64 (**)())(*(_QWORD *)v8 + 144LL);
  if ( v9 != sub_2C8F680 )
    v4 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, _QWORD))v9)(v8, a2, a3, a4, 0);
  v10 = *v7;
  v11 = *(_WORD *)(*v7 + 2) & 8;
  if ( v11 )
  {
    v36 = v4;
    v11 = 0;
    v14 = 0;
    v24 = sub_B2E500(v10);
    v25 = v36;
    v26 = v24;
    v27 = *(_QWORD *)v36;
    v28 = *(__int64 (**)())(*(_QWORD *)v36 + 872LL);
    if ( v28 != sub_2E2F9C0 )
    {
      v35 = ((__int64 (__fastcall *)(__int64))v28)(v36);
      v25 = v36;
      v11 = v35;
      v14 = v35;
      v27 = *(_QWORD *)v36;
    }
    v29 = *(__int64 (**)())(v27 + 880);
    v12 = 0;
    v13 = 0;
    if ( v29 != sub_2E2F9D0 )
    {
      v12 = ((__int64 (__fastcall *)(__int64, __int64))v29)(v25, v26);
      v13 = v12;
    }
  }
  else
  {
    v12 = 0;
    v13 = 0;
    v14 = 0;
  }
  *(_WORD *)a1 = v11;
  v15 = *(__int64 **)(a2 + 112);
  *(_WORD *)(a1 + 2) = v12;
  v16 = *(unsigned int *)(a2 + 120);
  *(_QWORD *)(a1 + 8) = v15;
  v17 = &v15[v16];
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 16) = v17;
  if ( v15 != v17 )
  {
    v18 = (_DWORD *)sub_2E33140(*v15);
    v19 = *v15;
    *(_QWORD *)(a1 + 24) = v18;
    if ( *(_DWORD **)(v19 + 192) == v18 )
    {
      v21 = v15 + 1;
      while ( 1 )
      {
        v23 = v21;
        if ( v17 == v21 )
          break;
        v22 = *v21++;
        if ( *(_QWORD *)(v22 + 192) != *(_QWORD *)(v22 + 184) )
        {
          *(_QWORD *)(a1 + 8) = v23;
          v18 = (_DWORD *)sub_2E33140(v22);
          *(_QWORD *)(a1 + 24) = v18;
          goto LABEL_7;
        }
      }
    }
    else
    {
LABEL_7:
      if ( *v18 != v13 && *v18 != v14 )
        return a1;
      v30 = *(__int64 **)(a1 + 8);
      v31 = *v30;
      while ( 1 )
      {
        v32 = v18 + 6;
        *(_QWORD *)(a1 + 24) = v32;
        if ( v32 == *(_DWORD **)(v31 + 192) )
          break;
LABEL_20:
        if ( *(_BYTE *)(v31 + 216) )
        {
          v18 = *(_DWORD **)(a1 + 24);
          if ( *v18 == v14 || *v18 == v13 )
            continue;
        }
        return a1;
      }
      while ( 1 )
      {
        v34 = v30++;
        if ( v17 == v30 )
          break;
        v33 = *v30;
        if ( *(_QWORD *)(*v30 + 192) != *(_QWORD *)(*v30 + 184) )
        {
          v37 = v34;
          *(_QWORD *)(a1 + 8) = v30;
          *(_QWORD *)(a1 + 24) = sub_2E33140(v33);
          v31 = v37[1];
          goto LABEL_20;
        }
      }
    }
    *(_QWORD *)(a1 + 8) = v17;
  }
  return a1;
}
