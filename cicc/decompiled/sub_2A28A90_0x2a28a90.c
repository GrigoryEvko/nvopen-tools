// Function: sub_2A28A90
// Address: 0x2a28a90
//
__int64 *__fastcall sub_2A28A90(__int64 *a1, __int64 a2)
{
  __int64 *v2; // r12
  __int64 v3; // rax
  __int64 *v4; // rbx
  __int64 v5; // r14
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // r13
  _QWORD *v10; // r15
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r15
  __int64 *v15; // rbx
  __int64 v16; // rdi
  __int64 v17; // r12
  __int64 v18; // rsi
  _QWORD *v19; // rax
  _QWORD *v20; // rdx
  __int64 *v21; // rbx
  const char *v22; // r15
  __int64 v23; // rdi
  __int64 v24; // r15
  int v25; // eax
  int v26; // eax
  unsigned int v27; // edx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 *result; // rax
  __int64 v32; // rbx
  __int64 v33; // r13
  __int64 v34; // rax
  __int64 v35; // rsi
  unsigned int v36; // ecx
  unsigned __int64 v37; // rdx
  __int64 v38; // r8
  __int64 v39; // r14
  int v40; // eax
  int v41; // eax
  unsigned int v42; // edx
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // rax
  unsigned __int64 v47; // rdx
  int v48; // edx
  int v49; // r9d
  __int64 *v50; // [rsp+8h] [rbp-B8h]
  __int64 v51; // [rsp+18h] [rbp-A8h]
  __int64 v52; // [rsp+20h] [rbp-A0h]
  __int64 v53; // [rsp+30h] [rbp-90h]
  __int64 *v54; // [rsp+38h] [rbp-88h]
  char *v55; // [rsp+40h] [rbp-80h] BYREF
  __int64 v56; // [rsp+48h] [rbp-78h]
  _QWORD v57[2]; // [rsp+50h] [rbp-70h] BYREF
  __int16 v58; // [rsp+60h] [rbp-60h]

  v2 = a1;
  v3 = sub_D47470(*a1);
  v4 = *(__int64 **)a2;
  v54 = (__int64 *)v3;
  v53 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 == v53 )
    goto LABEL_30;
LABEL_2:
  while ( 2 )
  {
    v5 = *v4;
    v6 = v54[7];
    if ( !v6 )
LABEL_62:
      BUG();
    if ( *(_BYTE *)(v6 - 24) != 84 )
    {
LABEL_7:
      v55 = (char *)sub_BD5D20(*v4);
      v58 = 773;
      v56 = v7;
      v57[0] = ".lver";
      v52 = *(_QWORD *)(v5 + 8);
      v8 = sub_BD2DA0(80);
      v9 = v8;
      if ( v8 )
      {
        v10 = (_QWORD *)v8;
        sub_B44260(v8, v52, 55, 0x8000000u, 0, 0);
        *(_DWORD *)(v9 + 72) = 2;
        sub_BD6B50((unsigned __int8 *)v9, (const char **)&v55);
        sub_BD2A10(v9, *(_DWORD *)(v9 + 72), 1);
      }
      else
      {
        v10 = 0;
      }
      v11 = v51;
      LOWORD(v11) = 1;
      v51 = v11;
      sub_B44220(v10, v54[7], v11);
      v55 = (char *)v57;
      v56 = 0x800000000LL;
      v14 = *(_QWORD *)(v5 + 16);
      if ( !v14 )
        goto LABEL_21;
      v50 = v4;
      v15 = v2;
      while ( 1 )
      {
        v16 = *v15;
        v17 = *(_QWORD *)(v14 + 24);
        v18 = *(_QWORD *)(v17 + 40);
        if ( *(_BYTE *)(*v15 + 84) )
        {
          v19 = *(_QWORD **)(v16 + 64);
          v20 = &v19[*(unsigned int *)(v16 + 76)];
          if ( v19 == v20 )
            goto LABEL_49;
          while ( v18 != *v19 )
          {
            if ( v20 == ++v19 )
              goto LABEL_49;
          }
LABEL_16:
          v14 = *(_QWORD *)(v14 + 8);
          if ( !v14 )
            goto LABEL_17;
        }
        else
        {
          if ( sub_C8CA60(v16 + 56, v18) )
            goto LABEL_16;
LABEL_49:
          v46 = (unsigned int)v56;
          v47 = (unsigned int)v56 + 1LL;
          if ( v47 > HIDWORD(v56) )
          {
            sub_C8D5F0((__int64)&v55, v57, v47, 8u, v12, v13);
            v46 = (unsigned int)v56;
          }
          *(_QWORD *)&v55[8 * v46] = v17;
          LODWORD(v56) = v56 + 1;
          v14 = *(_QWORD *)(v14 + 8);
          if ( !v14 )
          {
LABEL_17:
            v2 = v15;
            v4 = v50;
            if ( &v55[8 * (unsigned int)v56] != v55 )
            {
              v21 = (__int64 *)v55;
              v22 = &v55[8 * (unsigned int)v56];
              do
              {
                v23 = *v21++;
                sub_BD2ED0(v23, v5, v9);
              }
              while ( v22 != (const char *)v21 );
              v4 = v50;
            }
LABEL_21:
            v24 = sub_D46F00(*v2);
            v25 = *(_DWORD *)(v9 + 4) & 0x7FFFFFF;
            if ( v25 == *(_DWORD *)(v9 + 72) )
            {
              sub_B48D90(v9);
              v25 = *(_DWORD *)(v9 + 4) & 0x7FFFFFF;
            }
            v26 = (v25 + 1) & 0x7FFFFFF;
            v27 = v26 | *(_DWORD *)(v9 + 4) & 0xF8000000;
            v28 = *(_QWORD *)(v9 - 8) + 32LL * (unsigned int)(v26 - 1);
            *(_DWORD *)(v9 + 4) = v27;
            if ( *(_QWORD *)v28 )
            {
              v29 = *(_QWORD *)(v28 + 8);
              **(_QWORD **)(v28 + 16) = v29;
              if ( v29 )
                *(_QWORD *)(v29 + 16) = *(_QWORD *)(v28 + 16);
            }
            *(_QWORD *)v28 = v5;
            v30 = *(_QWORD *)(v5 + 16);
            *(_QWORD *)(v28 + 8) = v30;
            if ( v30 )
              *(_QWORD *)(v30 + 16) = v28 + 8;
            *(_QWORD *)(v28 + 16) = v5 + 16;
            *(_QWORD *)(v5 + 16) = v28;
            *(_QWORD *)(*(_QWORD *)(v9 - 8)
                      + 32LL * *(unsigned int *)(v9 + 72)
                      + 8LL * ((*(_DWORD *)(v9 + 4) & 0x7FFFFFFu) - 1)) = v24;
            if ( v55 == (char *)v57 )
              goto LABEL_54;
            _libc_free((unsigned __int64)v55);
            if ( (__int64 *)v53 == ++v4 )
              goto LABEL_30;
            goto LABEL_2;
          }
        }
      }
    }
    while ( v5 != **(_QWORD **)(v6 - 32) )
    {
      v6 = *(_QWORD *)(v6 + 8);
      if ( !v6 )
        goto LABEL_62;
      if ( *(_BYTE *)(v6 - 24) != 84 )
        goto LABEL_7;
    }
    sub_DACA20(v2[38], *v2, v6 - 24);
LABEL_54:
    if ( (__int64 *)v53 != ++v4 )
      continue;
    break;
  }
LABEL_30:
  result = v54;
  v32 = v54[7];
  if ( !v32 )
LABEL_61:
    BUG();
  if ( *(_BYTE *)(v32 - 24) == 84 )
  {
    while ( 1 )
    {
      v33 = **(_QWORD **)(v32 - 32);
      v34 = *((unsigned int *)v2 + 10);
      if ( (_DWORD)v34 )
      {
        v35 = v2[3];
        v36 = (v34 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
        v37 = v35 + ((unsigned __int64)v36 << 6);
        v38 = *(_QWORD *)(v37 + 24);
        if ( v33 == v38 )
        {
LABEL_34:
          if ( v37 != v35 + (v34 << 6) )
            v33 = *(_QWORD *)(v37 + 56);
        }
        else
        {
          v48 = 1;
          while ( v38 != -4096 )
          {
            v49 = v48 + 1;
            v36 = (v34 - 1) & (v48 + v36);
            v37 = v35 + ((unsigned __int64)v36 << 6);
            v38 = *(_QWORD *)(v37 + 24);
            if ( v33 == v38 )
              goto LABEL_34;
            v48 = v49;
          }
        }
      }
      v39 = sub_D46F00(v2[1]);
      v40 = *(_DWORD *)(v32 - 20) & 0x7FFFFFF;
      if ( v40 == *(_DWORD *)(v32 + 48) )
      {
        sub_B48D90(v32 - 24);
        v40 = *(_DWORD *)(v32 - 20) & 0x7FFFFFF;
      }
      v41 = (v40 + 1) & 0x7FFFFFF;
      v42 = v41 | *(_DWORD *)(v32 - 20) & 0xF8000000;
      v43 = *(_QWORD *)(v32 - 32) + 32LL * (unsigned int)(v41 - 1);
      *(_DWORD *)(v32 - 20) = v42;
      if ( *(_QWORD *)v43 )
      {
        v44 = *(_QWORD *)(v43 + 8);
        **(_QWORD **)(v43 + 16) = v44;
        if ( v44 )
          *(_QWORD *)(v44 + 16) = *(_QWORD *)(v43 + 16);
      }
      *(_QWORD *)v43 = v33;
      if ( v33 )
      {
        v45 = *(_QWORD *)(v33 + 16);
        *(_QWORD *)(v43 + 8) = v45;
        if ( v45 )
          *(_QWORD *)(v45 + 16) = v43 + 8;
        *(_QWORD *)(v43 + 16) = v33 + 16;
        *(_QWORD *)(v33 + 16) = v43;
      }
      result = (__int64 *)(*(_QWORD *)(v32 - 32)
                         + 32LL * *(unsigned int *)(v32 + 48)
                         + 8LL * ((*(_DWORD *)(v32 - 20) & 0x7FFFFFFu) - 1));
      *result = v39;
      v32 = *(_QWORD *)(v32 + 8);
      if ( !v32 )
        goto LABEL_61;
      if ( *(_BYTE *)(v32 - 24) != 84 )
        return result;
    }
  }
  return result;
}
