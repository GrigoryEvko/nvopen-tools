// Function: sub_1E13AC0
// Address: 0x1e13ac0
//
unsigned __int64 __fastcall sub_1E13AC0(__int64 *a1, int a2, __int64 a3)
{
  __int64 v4; // rsi
  __int64 v5; // rdx
  __int64 v6; // r8
  char v7; // r15
  __int64 v8; // rax
  int v9; // ecx
  __int64 v10; // r10
  unsigned int v11; // r11d
  _WORD *v12; // rbx
  unsigned __int16 v13; // r9
  __int16 *v14; // r11
  unsigned int v15; // r13d
  _WORD *v16; // rbx
  int v17; // r10d
  unsigned __int16 *v18; // r12
  unsigned int i; // ebx
  bool v20; // cf
  __int16 *v21; // r13
  __int16 v22; // r11
  __int16 *v23; // r9
  __int16 *v24; // rbx
  __int16 v25; // r10
  __int16 *v26; // r9
  unsigned __int16 v27; // r10
  __int16 v28; // r11
  __int16 *v29; // r9
  char v30; // r11
  __int64 v31; // rdx
  __int64 v32; // rcx
  unsigned __int16 v33; // ax
  int v35; // ebx
  unsigned __int8 v36; // cl
  char v37; // r10
  char v38; // bl
  char v39; // bl
  char v40; // bl
  __int64 v41; // [rsp+0h] [rbp-68h]
  __int64 v42; // [rsp+8h] [rbp-60h]
  __int64 v43; // [rsp+10h] [rbp-58h]
  unsigned __int8 v45; // [rsp+20h] [rbp-48h]
  __int64 v46; // [rsp+28h] [rbp-40h]
  unsigned __int8 v47; // [rsp+30h] [rbp-38h]
  char v48; // [rsp+31h] [rbp-37h]
  unsigned __int8 v49; // [rsp+32h] [rbp-36h]
  unsigned __int8 v50; // [rsp+33h] [rbp-35h]

  v4 = a1[3];
  v5 = a1[2];
  if ( v5 == v4 )
  {
    v45 = 0;
    v7 = 0;
    v31 = 0;
    v32 = 0;
    v50 = 0;
    LOBYTE(v46) = 0;
    v47 = 0;
    v49 = 0;
    goto LABEL_30;
  }
  v47 = 0;
  v6 = a1[1];
  v45 = 0;
  v7 = 0;
  v50 = 0;
  v41 = 24LL * (unsigned int)a2;
  v49 = 0;
  v48 = 1;
  LOBYTE(v46) = 0;
  v8 = *a1;
  while ( 1 )
  {
LABEL_3:
    if ( *(_BYTE *)v5 == 12 )
    {
      if ( ((*(_DWORD *)(*(_QWORD *)(v5 + 24) + 4LL * ((unsigned int)a2 >> 5)) >> (a2 & 0x1F)) & 1) == 0 )
        v7 = 1;
      goto LABEL_23;
    }
    if ( !*(_BYTE *)v5 )
    {
      v9 = *(_DWORD *)(v5 + 8);
      if ( v9 > 0 )
      {
        if ( a2 == v9 )
        {
LABEL_33:
          v30 = 1;
        }
        else
        {
          if ( a2 < 0 )
            goto LABEL_23;
          v10 = *(_QWORD *)(a3 + 8);
          v42 = *(_QWORD *)(a3 + 56);
          v43 = v41 + v10;
          v11 = *(_DWORD *)(v10 + 24LL * (unsigned int)v9 + 16);
          v12 = (_WORD *)(v42 + 2LL * (v11 >> 4));
          v13 = *v12 + v9 * (v11 & 0xF);
          v14 = v12 + 1;
          LODWORD(v12) = *(_DWORD *)(v41 + v10 + 16);
          v17 = a2 * ((unsigned __int8)v12 & 0xF);
          v15 = v13;
          v16 = (_WORD *)(v42 + 2LL * ((unsigned int)v12 >> 4));
          LOWORD(v17) = *v16 + v17;
          v18 = v16 + 1;
          for ( i = (unsigned __int16)v17; ; i = (unsigned __int16)v17 )
          {
            v20 = v15 < i;
            if ( v15 == i )
              break;
            while ( v20 )
            {
              v21 = v14 + 1;
              v22 = *v14;
              v13 += v22;
              if ( !v22 )
                goto LABEL_23;
              v14 = v21;
              v15 = v13;
              v20 = v13 < i;
              if ( v13 == i )
                goto LABEL_13;
            }
            v35 = *v18;
            if ( !(_WORD)v35 )
              goto LABEL_23;
            v17 += v35;
            ++v18;
          }
LABEL_13:
          v23 = (__int16 *)(v42 + 2LL * *(unsigned int *)(v43 + 8));
          v24 = 0;
          v25 = *v23;
          v26 = v23 + 1;
          if ( v25 )
            v24 = v26;
          v27 = a2 + v25;
LABEL_18:
          v29 = v24;
          while ( v29 )
          {
            if ( v9 == v27 )
              goto LABEL_33;
            v28 = *v29;
            v24 = 0;
            ++v29;
            v27 += v28;
            if ( !v28 )
              goto LABEL_18;
          }
          v30 = 0;
        }
        v36 = *(_BYTE *)(v5 + 3);
        v37 = v36 & 0x10;
        if ( (*(_BYTE *)(v5 + 4) & 1) != 0 || (*(_BYTE *)(v5 + 4) & 2) != 0 )
        {
          if ( v37 )
            goto LABEL_36;
        }
        else
        {
          if ( v37 && (*(_DWORD *)v5 & 0xFFF00) == 0 )
          {
LABEL_36:
            v38 = v50;
            v45 = 1;
            if ( v30 )
              v38 = 1;
            v50 = v38;
            v39 = v48;
            if ( (((v36 & 0x10) != 0) & (v36 >> 6)) == 0 )
              v39 = 0;
            v48 = v39;
            goto LABEL_23;
          }
          v49 = 1;
          if ( v30 )
          {
            v40 = v46;
            v47 = 1;
            if ( ((v37 == 0) & (v36 >> 6)) != 0 )
              v40 = 1;
            LOBYTE(v46) = v40;
          }
        }
      }
    }
LABEL_23:
    v5 += 40;
    a1[2] = v5;
    if ( v5 == v4 )
    {
      while ( 1 )
      {
        v8 = *(_QWORD *)(v8 + 8);
        *a1 = v8;
        if ( v6 == v8 || (*(_BYTE *)(v8 + 46) & 4) == 0 )
          break;
        v5 = *(_QWORD *)(v8 + 32);
        a1[2] = v5;
        v4 = v5 + 40LL * *(unsigned int *)(v8 + 40);
        a1[3] = v4;
        if ( v5 != v4 )
          goto LABEL_3;
      }
      v5 = a1[2];
      v4 = a1[3];
      if ( v4 == v5 )
        break;
    }
  }
  v31 = 0;
  v32 = 0;
  if ( v48 )
  {
    v32 = 1;
    if ( !v50 && !v7 )
    {
      v31 = v45;
      v32 = 0;
    }
  }
LABEL_30:
  LOBYTE(v33) = v7;
  HIBYTE(v33) = v45;
  return (v46 << 56)
       | (v31 << 48) & 0xFFFFFFFFFFFFFFLL
       | (v32 << 40) & 0xFFFFFFFFFFFFLL
       | ((unsigned __int64)v47 << 32) & 0xFFFFFFFFFFLL
       | (v49 << 24)
       | ((unsigned __int64)v50 << 16) & 0xFFFFFF
       | v33;
}
