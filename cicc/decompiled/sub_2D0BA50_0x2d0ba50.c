// Function: sub_2D0BA50
// Address: 0x2d0ba50
//
__int64 __fastcall sub_2D0BA50(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _BYTE *v6; // rsi
  unsigned int v7; // r8d
  __int64 v8; // r11
  int v9; // r10d
  __int64 v10; // r12
  unsigned int v11; // r13d
  unsigned int v12; // edi
  _QWORD *v13; // rcx
  __int64 v14; // rdx
  _QWORD *v15; // r9
  int v16; // edx
  __int64 v17; // rax
  unsigned int v18; // esi
  __int64 v19; // rcx
  unsigned int v20; // edx
  __int64 *v21; // rax
  __int64 v22; // r8
  int *v23; // rax
  int v24; // r13d
  __int64 v25; // r12
  __int64 v26; // rdx
  __int64 v27; // rax
  unsigned int v28; // r12d
  int v30; // eax
  unsigned int v31; // eax
  __int64 v32; // rdi
  int v33; // esi
  _QWORD *v34; // rcx
  _QWORD *v35; // rsi
  unsigned int v36; // r8d
  __int64 v37; // rcx
  int v38; // eax
  int v39; // r9d
  int v41; // [rsp+14h] [rbp-9Ch]
  int v42; // [rsp+18h] [rbp-98h]
  int v43; // [rsp+28h] [rbp-88h]
  int v44; // [rsp+2Ch] [rbp-84h]
  float v45; // [rsp+2Ch] [rbp-84h]
  __int64 v46; // [rsp+30h] [rbp-80h] BYREF
  __int64 v47; // [rsp+38h] [rbp-78h] BYREF
  _BYTE *v48; // [rsp+40h] [rbp-70h] BYREF
  _BYTE *v49; // [rsp+48h] [rbp-68h]
  _BYTE *v50; // [rsp+50h] [rbp-60h]
  __int64 v51; // [rsp+60h] [rbp-50h] BYREF
  __int64 v52; // [rsp+68h] [rbp-48h]
  __int64 v53; // [rsp+70h] [rbp-40h]
  __int64 v54; // [rsp+78h] [rbp-38h]

  v46 = a3;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  sub_9319A0((__int64)&v48, 0, &v46);
  v6 = v49;
LABEL_2:
  v7 = v54;
  v8 = v52;
  v9 = v54 - 1;
LABEL_3:
  while ( v6 != v48 )
  {
    v10 = *((_QWORD *)v6 - 1);
    v6 -= 8;
    v49 = v6;
    if ( a2 != v10 )
    {
      if ( !(_DWORD)v54 )
      {
        ++v51;
        goto LABEL_41;
      }
      v11 = ((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4);
      v12 = v9 & v11;
      v13 = (_QWORD *)(v52 + 8LL * (v9 & v11));
      v14 = *v13;
      if ( v10 != *v13 )
      {
        v44 = 1;
        v15 = 0;
        while ( v14 != -4096 )
        {
          if ( v15 || v14 != -8192 )
            v13 = v15;
          v12 = v9 & (v44 + v12);
          v14 = *(_QWORD *)(v52 + 8LL * v12);
          if ( v10 == v14 )
            goto LABEL_3;
          ++v44;
          v15 = v13;
          v13 = (_QWORD *)(v52 + 8LL * v12);
        }
        if ( !v15 )
          v15 = v13;
        ++v51;
        v16 = v53 + 1;
        if ( 4 * ((int)v53 + 1) < (unsigned int)(3 * v54) )
        {
          if ( (int)v54 - HIDWORD(v53) - v16 > (unsigned int)v54 >> 3 )
            goto LABEL_13;
          sub_CF28B0((__int64)&v51, v54);
          if ( (_DWORD)v54 )
          {
            v35 = 0;
            v36 = (v54 - 1) & v11;
            v15 = (_QWORD *)(v52 + 8LL * v36);
            v37 = *v15;
            v16 = v53 + 1;
            v38 = 1;
            if ( v10 != *v15 )
            {
              while ( v37 != -4096 )
              {
                if ( v37 == -8192 && !v35 )
                  v35 = v15;
                v36 = (v54 - 1) & (v38 + v36);
                v15 = (_QWORD *)(v52 + 8LL * v36);
                v37 = *v15;
                if ( v10 == *v15 )
                  goto LABEL_13;
                ++v38;
              }
              if ( v35 )
                v15 = v35;
            }
            goto LABEL_13;
          }
LABEL_71:
          LODWORD(v53) = v53 + 1;
          BUG();
        }
LABEL_41:
        sub_CF28B0((__int64)&v51, 2 * v54);
        if ( !(_DWORD)v54 )
          goto LABEL_71;
        v31 = (v54 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v15 = (_QWORD *)(v52 + 8LL * v31);
        v16 = v53 + 1;
        v32 = *v15;
        if ( v10 != *v15 )
        {
          v33 = 1;
          v34 = 0;
          while ( v32 != -4096 )
          {
            if ( v32 == -8192 && !v34 )
              v34 = v15;
            v31 = (v54 - 1) & (v33 + v31);
            v15 = (_QWORD *)(v52 + 8LL * v31);
            v32 = *v15;
            if ( v10 == *v15 )
              goto LABEL_13;
            ++v33;
          }
          if ( v34 )
            v15 = v34;
        }
LABEL_13:
        LODWORD(v53) = v16;
        if ( *v15 != -4096 )
          --HIDWORD(v53);
        *v15 = v10;
        v17 = *(_QWORD *)(a1 + 48);
        v18 = *(_DWORD *)(v17 + 136);
        v19 = *(_QWORD *)(v17 + 120);
        if ( v18 )
        {
          v20 = (v18 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v21 = (__int64 *)(v19 + 16LL * v20);
          v22 = *v21;
          if ( v10 == *v21 )
            goto LABEL_17;
          v30 = 1;
          while ( v22 != -4096 )
          {
            v39 = v30 + 1;
            v20 = (v18 - 1) & (v30 + v20);
            v21 = (__int64 *)(v19 + 16LL * v20);
            v22 = *v21;
            if ( v10 == *v21 )
              goto LABEL_17;
            v30 = v39;
          }
        }
        v21 = (__int64 *)(v19 + 16LL * v18);
LABEL_17:
        v23 = (int *)v21[1];
        v24 = v23[4];
        v45 = *(float *)&qword_5015DE8;
        v42 = v23[5];
        v43 = *v23;
        v41 = v23[1];
        if ( !sub_BCAC40(*(_QWORD *)(a4 + 8), 1) )
        {
          if ( (float)v43 <= (float)((float)v24 * v45) )
            goto LABEL_19;
LABEL_32:
          v7 = v54;
          v8 = v52;
          v28 = 1;
          goto LABEL_34;
        }
        if ( (float)v41 > (float)((float)v42 * v45) )
          goto LABEL_32;
LABEL_19:
        v25 = *(_QWORD *)(v10 + 16);
        v6 = v49;
        if ( v25 )
        {
          while ( 1 )
          {
            v26 = *(_QWORD *)(v25 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v26 - 30) <= 0xAu )
              break;
            v25 = *(_QWORD *)(v25 + 8);
            if ( !v25 )
              goto LABEL_2;
          }
LABEL_27:
          v27 = *(_QWORD *)(v26 + 40);
          v47 = v27;
          if ( v6 == v50 )
          {
            sub_9319A0((__int64)&v48, v6, &v47);
            v6 = v49;
          }
          else
          {
            if ( v6 )
            {
              *(_QWORD *)v6 = v27;
              v6 = v49;
            }
            v6 += 8;
            v49 = v6;
          }
          while ( 1 )
          {
            v25 = *(_QWORD *)(v25 + 8);
            if ( !v25 )
              break;
            v26 = *(_QWORD *)(v25 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v26 - 30) <= 0xAu )
              goto LABEL_27;
          }
        }
        goto LABEL_2;
      }
    }
  }
  v28 = 0;
LABEL_34:
  sub_C7D6A0(v8, 8LL * v7, 8);
  if ( v48 )
    j_j___libc_free_0((unsigned __int64)v48);
  return v28;
}
