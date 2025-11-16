// Function: sub_2E9BF00
// Address: 0x2e9bf00
//
void __fastcall sub_2E9BF00(__int64 *a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  char *v7; // rbx
  char *v8; // r8
  __int64 *v10; // r14
  char v11; // r12
  __int64 *v12; // rax
  __int64 *v13; // rdx
  __int64 v14; // rsi
  char v15; // al
  __int64 v16; // rdx
  unsigned __int8 v17; // al
  __int64 v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 *v21; // r11
  __int64 v22; // r10
  __int64 v23; // rsi
  int v24; // eax
  unsigned int v25; // edx
  int v26; // edi
  int v27; // eax
  __int64 v28; // rsi
  __int64 v29; // r10
  __int16 *v30; // rsi
  __int64 v31; // rax
  int v32; // edx
  __int64 (*v33)(); // rax
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rbx
  __int64 v37; // rbx
  __int64 v38; // rax
  __int64 *v39; // rax
  int v40; // eax
  __int64 v41; // rax
  __int64 v42; // rax
  int *v43; // rdx
  int v44; // eax
  __int64 v45; // rsi
  __int64 **v46; // rax
  __int64 **v47; // rsi
  __int64 v48; // rdx
  unsigned __int64 v49; // rdx
  __int64 v50; // [rsp+8h] [rbp-A8h]
  char *v51; // [rsp+10h] [rbp-A0h]
  char *v52; // [rsp+10h] [rbp-A0h]
  char v54; // [rsp+20h] [rbp-90h]
  char *v55; // [rsp+20h] [rbp-90h]
  unsigned int v56; // [rsp+20h] [rbp-90h]
  unsigned int v58; // [rsp+34h] [rbp-7Ch]
  int v60; // [rsp+4Ch] [rbp-64h] BYREF
  _DWORD v61[24]; // [rsp+50h] [rbp-60h] BYREF

  v7 = *(char **)(a2 + 32);
  v50 = a6;
  v8 = &v7[40 * (*(_DWORD *)(a2 + 40) & 0xFFFFFF)];
  if ( v8 == v7 )
    return;
  v10 = (__int64 *)a4;
  v11 = 0;
  v54 = 0;
  v58 = 0;
  do
  {
    while ( 1 )
    {
      v15 = *v7;
      if ( *v7 == 5 )
        break;
      if ( v15 == 12 )
      {
        v51 = v8;
        sub_2E98800(a1[2], (__int64)v10, *((int **)v7 + 3), a4, (__int64)v8, a6);
        v8 = v51;
      }
      else if ( !v15 )
      {
        v16 = *((unsigned int *)v7 + 2);
        if ( (_DWORD)v16 )
        {
          v17 = v7[3];
          if ( (v17 & 0x10) != 0 )
          {
            v18 = a1[2];
            a6 = *(_QWORD *)(v18 + 8);
            a4 = *(_DWORD *)(a6 + 24LL * (unsigned int)v16 + 16) & 0xFFF;
            v14 = *(_QWORD *)(v18 + 56) + 2LL * (*(_DWORD *)(a6 + 24LL * (unsigned int)v16 + 16) >> 12);
            if ( (v17 & 0x20) != 0 )
            {
              if ( v14 )
              {
                do
                {
                  v14 += 2;
                  *(_QWORD *)(*v10 + 8LL * ((unsigned int)a4 >> 6)) |= 1LL << a4;
                  a4 = (unsigned int)(*(__int16 *)(v14 - 2) + (_DWORD)a4);
                }
                while ( *(_WORD *)(v14 - 2) );
                v17 = v7[3];
              }
              if ( (((v17 & 0x10) != 0) & (v17 >> 6)) == 0 )
                v11 = 1;
            }
            else
            {
              if ( v58 )
                v11 = 1;
              else
                v58 = *((_DWORD *)v7 + 2);
              if ( v14 )
              {
                do
                {
                  v19 = 1LL << a4;
                  v20 = 8LL * ((unsigned int)a4 >> 6);
                  v21 = (__int64 *)(v20 + *v10);
                  v13 = (__int64 *)(v20 + *a3);
                  v22 = *v21;
                  a6 = *v13;
                  if ( (*v13 & (1LL << a4)) != 0 )
                  {
                    v11 = 1;
                    *v21 = v19 | v22;
                    v12 = (__int64 *)(*a3 + v20);
                    a6 = *v12;
                    v13 = v12;
                  }
                  else if ( (v19 & v22) != 0 )
                  {
                    v11 = 1;
                  }
                  v14 += 2;
                  *v13 = a6 | v19;
                  a4 = (unsigned int)(*(__int16 *)(v14 - 2) + (_DWORD)a4);
                }
                while ( *(_WORD *)(v14 - 2) );
              }
            }
            goto LABEL_5;
          }
          if ( !v54 )
          {
            v28 = a1[2];
            v29 = *(_QWORD *)(v28 + 8);
            v30 = (__int16 *)(*(_QWORD *)(v28 + 56) + 2LL * (*(_DWORD *)(v29 + 24 * v16 + 16) >> 12));
            a4 = *(_DWORD *)(v29 + 24 * v16 + 16) & 0xFFF;
            if ( v30 )
            {
              do
              {
                v31 = (unsigned int)a4 >> 6;
                a6 = *(_QWORD *)(*a3 + 8 * v31) & (1LL << a4);
                if ( a6 )
                  goto LABEL_22;
                a6 = *v10;
                if ( (*(_QWORD *)(*v10 + 8 * v31) & (1LL << a4)) != 0 )
                  goto LABEL_22;
                v32 = *v30++;
                a4 = (unsigned int)(v32 + a4);
              }
              while ( (_WORD)v32 );
            }
          }
        }
      }
LABEL_5:
      v7 += 40;
      if ( v8 == v7 )
        goto LABEL_23;
    }
    a4 = *((unsigned int *)v7 + 6);
    v60 = *((_DWORD *)v7 + 6);
    if ( (*(_BYTE *)(a5 + 8) & 1) != 0 )
    {
      v23 = a5 + 16;
      v24 = 3;
    }
    else
    {
      v23 = *(_QWORD *)(a5 + 16);
      v27 = *(_DWORD *)(a5 + 24);
      if ( !v27 )
        goto LABEL_45;
      v24 = v27 - 1;
    }
    v25 = v24 & (37 * a4);
    v26 = *(_DWORD *)(v23 + 4LL * v25);
    if ( (_DWORD)a4 == v26 )
      goto LABEL_22;
    a6 = 1;
    while ( v26 != 0x7FFFFFFF )
    {
      v25 = v24 & (a6 + v25);
      v26 = *(_DWORD *)(v23 + 4LL * v25);
      if ( (_DWORD)a4 == v26 )
        goto LABEL_22;
      a6 = (unsigned int)(a6 + 1);
    }
LABEL_45:
    if ( *(_BYTE *)(*(_QWORD *)(a1[3] + 8) + 40LL * (unsigned int)(*(_DWORD *)(a1[3] + 32) + a4) + 18) )
    {
      if ( (unsigned int)*(unsigned __int16 *)(a2 + 68) - 1 <= 1
        && (*(_BYTE *)(*(_QWORD *)(a2 + 32) + 64LL) & 0x10) != 0
        || ((v40 = *(_DWORD *)(a2 + 44), (v40 & 4) == 0) && (v40 & 8) != 0
          ? (v52 = v8, v56 = a4, LOBYTE(v41) = sub_2E88A90(a2, 0x100000, 1), a4 = v56, v8 = v52)
          : (char *)(v41 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 20) & 1LL),
            (_BYTE)v41) )
      {
        v42 = *(_QWORD *)(a2 + 48);
        v43 = (int *)(v42 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (v42 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          goto LABEL_66;
        if ( (v42 & 7) != 0 )
        {
          if ( (v42 & 7) != 3 || !*v43 )
            goto LABEL_66;
        }
        else
        {
          *(_QWORD *)(a2 + 48) = v43;
          LOBYTE(v42) = v42 & 0xF8;
        }
        v44 = v42 & 7;
        if ( v44 )
        {
          if ( v44 != 3 )
            goto LABEL_22;
          v46 = (__int64 **)(v43 + 4);
          v45 = *v43;
        }
        else
        {
          v45 = 1;
          *(_QWORD *)(a2 + 48) = v43;
          v46 = (__int64 **)(a2 + 48);
        }
        v47 = &v46[v45];
        if ( v46 != v47 )
        {
          while ( 1 )
          {
            if ( ((*v46)[4] & 2) != 0 )
            {
              v48 = **v46;
              if ( v48 )
              {
                if ( (v48 & 4) != 0 )
                {
                  v49 = v48 & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v49 )
                  {
                    if ( *(_DWORD *)(v49 + 8) == 4 && (_DWORD)a4 == *(_DWORD *)(v49 + 16) )
                      break;
                  }
                }
              }
            }
            if ( v47 == ++v46 )
              goto LABEL_22;
          }
LABEL_66:
          v55 = v8;
          sub_2E9BB80((__int64)v61, a5, &v60);
          v8 = v55;
        }
      }
    }
LABEL_22:
    v7 += 40;
    v54 = 1;
  }
  while ( v8 != v7 );
LABEL_23:
  if ( v58 && !v11 )
  {
    v61[0] = 0x80000000;
    if ( v54 || !(unsigned __int8)sub_2E9AFF0((__int64)a1, a2, a7, a4, v8) )
    {
      v33 = *(__int64 (**)())(*(_QWORD *)*a1 + 88LL);
      if ( v33 == sub_2E97330 )
        return;
      if ( !((unsigned int (__fastcall *)(__int64, __int64, _DWORD *, __int64))v33)(*a1, a2, v61, a4) )
        return;
      v36 = v61[0];
      if ( !*(_BYTE *)(*(_QWORD *)(a1[3] + 8) + 40LL * (unsigned int)(*(_DWORD *)(a1[3] + 32) + v61[0]) + 18) )
        return;
    }
    else
    {
      v36 = v61[0];
    }
    v37 = (v36 << 32) | v58;
    v38 = *(unsigned int *)(v50 + 8);
    if ( v38 + 1 > (unsigned __int64)*(unsigned int *)(v50 + 12) )
    {
      sub_C8D5F0(v50, (const void *)(v50 + 16), v38 + 1, 0x10u, v34, v35);
      v38 = *(unsigned int *)(v50 + 8);
    }
    v39 = (__int64 *)(*(_QWORD *)v50 + 16 * v38);
    *v39 = a2;
    v39[1] = v37;
    ++*(_DWORD *)(v50 + 8);
  }
}
