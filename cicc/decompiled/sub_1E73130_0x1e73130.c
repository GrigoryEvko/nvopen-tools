// Function: sub_1E73130
// Address: 0x1e73130
//
_DWORD *__fastcall sub_1E73130(__int64 a1, __int64 a2)
{
  __int64 *v4; // rdi
  __int64 v5; // rax
  void (*v6)(void); // rdx
  void (*v7)(); // rax
  int v8; // eax
  __int64 v9; // rdi
  unsigned int v10; // r13d
  int v11; // edx
  unsigned int v12; // eax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rsi
  unsigned __int16 *v17; // r15
  unsigned __int16 *v18; // r14
  unsigned int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // rdi
  unsigned __int16 *v23; // r14
  unsigned __int16 *v24; // r8
  __int64 v25; // r15
  unsigned int *v26; // r15
  unsigned int *v27; // r14
  char v28; // al
  unsigned int v29; // edx
  unsigned int v30; // eax
  unsigned int v31; // eax
  __int64 v32; // rdx
  __int64 v33; // rdi
  int v34; // edx
  _DWORD *result; // rax
  unsigned int v36; // eax
  unsigned int v37; // r9d
  __int64 v38; // r13
  unsigned __int16 *v39; // [rsp+8h] [rbp-48h]
  __int64 v40; // [rsp+10h] [rbp-40h]
  unsigned int v41; // [rsp+10h] [rbp-40h]
  int v42; // [rsp+1Ch] [rbp-34h]

  v4 = *(__int64 **)(a1 + 152);
  if ( !*((_DWORD *)v4 + 2) )
    goto LABEL_7;
  v5 = *v4;
  if ( *(_DWORD *)(a1 + 24) != 1 && (*(_BYTE *)(a2 + 228) & 2) != 0 )
  {
    v6 = *(void (**)(void))(v5 + 32);
    if ( v6 != nullsub_678 )
    {
      v6();
      v4 = *(__int64 **)(a1 + 152);
      v5 = *v4;
    }
  }
  v7 = *(void (**)())(v5 + 40);
  if ( v7 != nullsub_679 )
  {
    ((void (__fastcall *)(__int64 *, __int64))v7)(v4, a2);
    v40 = *(_QWORD *)(a2 + 24);
    if ( v40 )
      goto LABEL_8;
  }
  else
  {
LABEL_7:
    v40 = *(_QWORD *)(a2 + 24);
    if ( v40 )
      goto LABEL_8;
  }
  v38 = *(_QWORD *)a1 + 632LL;
  if ( (unsigned __int8)sub_1F4B670(v38) )
  {
    v40 = sub_1F4B8B0(v38, *(_QWORD *)(a2 + 8));
    *(_QWORD *)(a2 + 24) = v40;
  }
  else
  {
    v40 = *(_QWORD *)(a2 + 24);
  }
LABEL_8:
  v8 = sub_1F4BA40(*(_QWORD *)(a1 + 8), *(_QWORD *)(a2 + 8), 0);
  v9 = *(_QWORD *)(a1 + 8);
  v42 = v8;
  v10 = *(_DWORD *)(a1 + 164);
  v11 = *(_DWORD *)(v9 + 4);
  v12 = *(_DWORD *)(a2 + 252);
  if ( *(_DWORD *)(a1 + 24) == 1 )
    v12 = *(_DWORD *)(a2 + 248);
  if ( v11 && (v11 == 1 || (*(_BYTE *)(a2 + 229) & 0x40) != 0) && v10 < v12 )
    v10 = v12;
  *(_DWORD *)(a1 + 184) += v42;
  if ( (unsigned __int8)sub_1F4B670(v9) )
  {
    *(_DWORD *)(*(_QWORD *)(a1 + 16) + 8LL) -= *(_DWORD *)(*(_QWORD *)(a1 + 8) + 272LL) * v42;
    v13 = *(unsigned int *)(a1 + 276);
    v14 = *(_QWORD *)(a1 + 8);
    if ( (_DWORD)v13
      && *(_DWORD *)(v14 + 272) * *(_DWORD *)(a1 + 184) - *(_DWORD *)(*(_QWORD *)(a1 + 192) + 4 * v13) >= *(_DWORD *)(v14 + 276) )
    {
      *(_DWORD *)(a1 + 276) = 0;
    }
    v15 = *(_QWORD *)(*(_QWORD *)(v14 + 176) + 136LL);
    v16 = *(unsigned __int16 *)(v40 + 2);
    v17 = (unsigned __int16 *)(v15 + 4 * v16);
    v18 = (unsigned __int16 *)(v15 + 4 * (v16 + *(unsigned __int16 *)(v40 + 4)));
    if ( v18 != v17 )
    {
      do
      {
        v19 = sub_1E73070(a1, *v17, v17[1]);
        if ( v10 < v19 )
          v10 = v19;
        v17 += 2;
      }
      while ( v17 != v18 );
      if ( *(char *)(a2 + 229) < 0 )
      {
        v20 = *(_QWORD *)(a1 + 8);
        v21 = *(_QWORD *)(*(_QWORD *)(v20 + 176) + 136LL);
        v22 = *(unsigned __int16 *)(v40 + 2);
        v23 = (unsigned __int16 *)(v21 + 4 * v22);
        v24 = (unsigned __int16 *)(v21 + 4 * (v22 + *(unsigned __int16 *)(v40 + 4)));
        if ( v24 != v23 )
        {
          while ( 1 )
          {
            v25 = *v23;
            if ( !*(_DWORD *)(*(_QWORD *)(v20 + 32) + 32 * v25 + 16) )
            {
              if ( *(_DWORD *)(a1 + 24) == 1 )
              {
                v39 = v24;
                v41 = v10 + v23[1];
                v36 = sub_1E72BE0(a1, *v23, 0);
                v37 = v41;
                v24 = v39;
                if ( v41 < v36 )
                  v37 = v36;
                *(_DWORD *)(*(_QWORD *)(a1 + 288) + 4 * v25) = v37;
              }
              else
              {
                *(_DWORD *)(*(_QWORD *)(a1 + 288) + 4 * v25) = v10;
              }
            }
            v23 += 2;
            if ( v23 == v24 )
              break;
            v20 = *(_QWORD *)(a1 + 8);
          }
        }
      }
    }
  }
  v26 = (unsigned int *)(a1 + 180);
  v27 = (unsigned int *)(a1 + 176);
  if ( *(_DWORD *)(a1 + 24) == 1 )
  {
    v27 = (unsigned int *)(a1 + 180);
    v26 = (unsigned int *)(a1 + 176);
  }
  v28 = *(_BYTE *)(a2 + 236);
  if ( (v28 & 1) != 0 )
  {
    v29 = *(_DWORD *)(a2 + 240);
    if ( *v26 >= v29 )
      goto LABEL_35;
  }
  else
  {
    sub_1F01DD0(a2);
    v29 = *(_DWORD *)(a2 + 240);
    if ( *v26 >= v29 )
      goto LABEL_34;
    if ( (*(_BYTE *)(a2 + 236) & 1) == 0 )
    {
      sub_1F01DD0(a2);
      v29 = *(_DWORD *)(a2 + 240);
    }
  }
  *v26 = v29;
LABEL_34:
  v28 = *(_BYTE *)(a2 + 236);
LABEL_35:
  if ( (v28 & 2) != 0 )
  {
    v30 = *(_DWORD *)(a2 + 244);
    if ( *v27 >= v30 )
      goto LABEL_38;
    goto LABEL_37;
  }
  sub_1F01F70(a2);
  v30 = *(_DWORD *)(a2 + 244);
  if ( *v27 < v30 )
  {
    if ( (*(_BYTE *)(a2 + 236) & 2) != 0 )
    {
LABEL_37:
      *v27 = v30;
      goto LABEL_38;
    }
    sub_1F01F70(a2);
    *v27 = *(_DWORD *)(a2 + 244);
  }
LABEL_38:
  v31 = *(_DWORD *)(a1 + 164);
  if ( v31 < v10 )
  {
    sub_1E72EF0(a1, v10);
    *(_DWORD *)(a1 + 168) += v42;
    v33 = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(a1 + 24) != 1 )
      goto LABEL_44;
  }
  else
  {
    v32 = *(unsigned int *)(a1 + 276);
    v33 = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(a1 + 176) >= v31 )
      v31 = *(_DWORD *)(a1 + 176);
    if ( (_DWORD)v32 )
      v34 = *(_DWORD *)(*(_QWORD *)(a1 + 192) + 4 * v32);
    else
      v34 = *(_DWORD *)(v33 + 272) * *(_DWORD *)(a1 + 184);
    *(_BYTE *)(a1 + 280) = (int)(v34 - *(_DWORD *)(v33 + 276) * v31) > *(_DWORD *)(v33 + 276);
    *(_DWORD *)(a1 + 168) += v42;
    if ( *(_DWORD *)(a1 + 24) != 1 )
    {
LABEL_44:
      if ( !(unsigned __int8)sub_1F4B960(v33, *(_QWORD *)(a2 + 8), 0) )
        goto LABEL_46;
      goto LABEL_45;
    }
  }
  if ( (unsigned __int8)sub_1F4B9D0(v33, *(_QWORD *)(a2 + 8), 0) )
  {
LABEL_45:
    sub_1E72EF0(a1, ++v10);
    goto LABEL_46;
  }
  if ( *(_DWORD *)(a1 + 24) != 1 )
  {
    v33 = *(_QWORD *)(a1 + 8);
    goto LABEL_44;
  }
LABEL_46:
  result = *(_DWORD **)(a1 + 8);
  if ( *result <= *(_DWORD *)(a1 + 168) )
  {
    do
    {
      sub_1E72EF0(a1, ++v10);
      result = (_DWORD *)**(unsigned int **)(a1 + 8);
    }
    while ( *(_DWORD *)(a1 + 168) >= (unsigned int)result );
  }
  return result;
}
