// Function: sub_3435EB0
// Address: 0x3435eb0
//
__int64 __fastcall sub_3435EB0(__int64 a1, __int64 a2, int a3)
{
  unsigned __int8 v4; // al
  __int64 v6; // rax
  unsigned __int8 *v7; // rax
  __int64 v8; // rbx
  unsigned int v9; // esi
  __int64 v10; // r9
  __int64 v11; // r11
  int v12; // r13d
  unsigned int v13; // ecx
  __int64 v14; // rdx
  unsigned __int8 *v15; // r8
  __int64 v16; // rsi
  __int64 v17; // rdx
  unsigned int v18; // ecx
  __int64 *v19; // rax
  __int64 v20; // r8
  int v21; // eax
  __int64 v22; // rax
  _QWORD *v23; // r14
  unsigned int v24; // ebx
  int v25; // r12d
  char v26; // r15
  int v27; // esi
  char v28; // al
  int v29; // eax
  int v30; // r10d
  int v31; // ecx
  int v32; // ecx
  _QWORD *v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+18h] [rbp-48h]
  unsigned __int8 *v35; // [rsp+20h] [rbp-40h] BYREF
  __int64 v36; // [rsp+28h] [rbp-38h] BYREF

  if ( !a3 )
    goto LABEL_4;
  while ( 1 )
  {
    v4 = *(_BYTE *)a1;
    if ( *(_BYTE *)a1 == 85 )
    {
      v6 = *(_QWORD *)(a1 - 32);
      if ( !v6 )
        goto LABEL_4;
      if ( *(_BYTE *)v6 )
        goto LABEL_4;
      if ( *(_QWORD *)(v6 + 24) != *(_QWORD *)(a1 + 80) )
        goto LABEL_4;
      if ( (*(_BYTE *)(v6 + 33) & 0x20) == 0 )
        goto LABEL_4;
      if ( *(_DWORD *)(v6 + 36) != 149 )
        goto LABEL_4;
      v7 = (unsigned __int8 *)sub_B5B6B0(a1);
      if ( (unsigned int)*v7 - 12 <= 1 )
        goto LABEL_4;
      v8 = *(_QWORD *)(a2 + 960);
      v35 = v7;
      v9 = *(_DWORD *)(v8 + 240);
      if ( v9 )
      {
        v10 = *(_QWORD *)(v8 + 224);
        v11 = 0;
        v12 = 1;
        v13 = (v9 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v14 = v10 + 40LL * v13;
        v15 = *(unsigned __int8 **)v14;
        if ( v7 == *(unsigned __int8 **)v14 )
        {
LABEL_14:
          v16 = *(_QWORD *)(v14 + 16);
          v17 = *(unsigned int *)(v14 + 32);
          if ( !(_DWORD)v17 )
            goto LABEL_4;
          v18 = (v17 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
          v19 = (__int64 *)(v16 + 16LL * v18);
          v20 = *v19;
          if ( *v19 == a1 )
          {
LABEL_16:
            if ( v19 != (__int64 *)(v16 + 16 * v17) && *((_DWORD *)v19 + 2) == 1 )
            {
              v21 = *((_DWORD *)v19 + 3);
              BYTE4(v36) = 1;
              LODWORD(v36) = v21;
              return v36;
            }
          }
          else
          {
            v29 = 1;
            while ( v20 != -4096 )
            {
              v30 = v29 + 1;
              v18 = (v17 - 1) & (v29 + v18);
              v19 = (__int64 *)(v16 + 16LL * v18);
              v20 = *v19;
              if ( a1 == *v19 )
                goto LABEL_16;
              v29 = v30;
            }
          }
LABEL_4:
          BYTE4(v36) = 0;
          return v36;
        }
        while ( v15 != (unsigned __int8 *)-4096LL )
        {
          if ( v15 == (unsigned __int8 *)-8192LL && !v11 )
            v11 = v14;
          v13 = (v9 - 1) & (v12 + v13);
          v14 = v10 + 40LL * v13;
          v15 = *(unsigned __int8 **)v14;
          if ( v7 == *(unsigned __int8 **)v14 )
            goto LABEL_14;
          ++v12;
        }
        if ( v11 )
          v14 = v11;
        v36 = v14;
        v31 = *(_DWORD *)(v8 + 232);
        ++*(_QWORD *)(v8 + 216);
        v32 = v31 + 1;
        if ( 4 * v32 < 3 * v9 )
        {
          if ( v9 - *(_DWORD *)(v8 + 236) - v32 > v9 >> 3 )
          {
LABEL_48:
            *(_DWORD *)(v8 + 232) = v32;
            if ( *(_QWORD *)v14 != -4096 )
              --*(_DWORD *)(v8 + 236);
            *(_QWORD *)v14 = v7;
            *(_QWORD *)(v14 + 8) = 0;
            *(_QWORD *)(v14 + 16) = 0;
            *(_QWORD *)(v14 + 24) = 0;
            *(_DWORD *)(v14 + 32) = 0;
            goto LABEL_4;
          }
LABEL_53:
          sub_3435C30(v8 + 216, v9);
          sub_3434540(v8 + 216, (__int64 *)&v35, &v36);
          v7 = v35;
          v14 = v36;
          v32 = *(_DWORD *)(v8 + 232) + 1;
          goto LABEL_48;
        }
      }
      else
      {
        v36 = 0;
        ++*(_QWORD *)(v8 + 216);
      }
      v9 *= 2;
      goto LABEL_53;
    }
    if ( v4 <= 0x1Cu )
      goto LABEL_4;
    if ( v4 != 78 )
      break;
    a1 = *(_QWORD *)(a1 - 32);
    if ( !--a3 )
      goto LABEL_4;
  }
  if ( v4 != 84 )
    goto LABEL_4;
  HIDWORD(v34) = 0;
  v22 = 4LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    v23 = *(_QWORD **)(a1 - 8);
    v33 = &v23[v22];
  }
  else
  {
    v33 = (_QWORD *)a1;
    v23 = (_QWORD *)(a1 - v22 * 8);
  }
  if ( v33 != v23 )
  {
    v24 = a3 - 1;
    v25 = 0;
    v26 = 0;
    while ( 1 )
    {
      v27 = v25;
      v35 = (unsigned __int8 *)sub_3435EB0(*v23, a2, v24);
      v25 = (int)v35;
      v28 = v26;
      v26 = BYTE4(v35);
      if ( !BYTE4(v35) || (_DWORD)v35 != v27 && v28 )
        goto LABEL_4;
      v23 += 4;
      HIDWORD(v34) = HIDWORD(v35);
      if ( v33 == v23 )
        goto LABEL_31;
    }
  }
  v25 = 0;
  v26 = 0;
LABEL_31:
  LODWORD(v34) = v25;
  BYTE4(v34) = v26;
  return v34;
}
