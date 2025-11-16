// Function: sub_D63C20
// Address: 0xd63c20
//
__int64 __fastcall sub_D63C20(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r12
  unsigned int v6; // eax
  __int64 v7; // rdx
  unsigned int v8; // eax
  __int64 v9; // rdx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  _QWORD *v13; // r13
  __int64 v14; // rdx
  _QWORD *v15; // r15
  __int64 *v16; // rax
  __int64 v17; // rdx
  __int64 *v18; // r13
  __int64 v19; // r14
  __int64 *v20; // r15
  __int64 *v21; // rax
  __int64 v22; // rax
  __int64 v23; // r9
  __int64 *v24; // r14
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  _QWORD *v29; // rax
  __int64 v30; // rax
  int v31; // r8d
  unsigned __int64 v32; // [rsp+10h] [rbp-50h] BYREF
  __int64 v33; // [rsp+18h] [rbp-48h]
  __int64 v34; // [rsp+20h] [rbp-40h]

  v3 = sub_AE4570(*(_QWORD *)a1, *(_QWORD *)(a2 + 8));
  *(_QWORD *)(a1 + 208) = v3;
  *(_QWORD *)(a1 + 216) = sub_ACD640(v3, 0, 0);
  v5 = sub_D63080(a1, (unsigned __int8 *)a2);
  if ( !v5 || !v4 )
  {
    v11 = *(_QWORD **)(a1 + 264);
    if ( *(_BYTE *)(a1 + 284) )
      v12 = *(unsigned int *)(a1 + 276);
    else
      v12 = *(unsigned int *)(a1 + 272);
    v13 = &v11[v12];
    if ( v11 != v13 )
    {
      while ( 1 )
      {
        v14 = *v11;
        v15 = v11;
        if ( *v11 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v13 == ++v11 )
          goto LABEL_21;
      }
      if ( v13 != v11 )
      {
        do
        {
          v22 = *(unsigned int *)(a1 + 248);
          v23 = *(_QWORD *)(a1 + 232);
          if ( (_DWORD)v22 )
          {
            a2 = ((_DWORD)v22 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
            v24 = (__int64 *)(v23 + 56 * a2);
            v25 = *v24;
            if ( v14 == *v24 )
            {
LABEL_36:
              if ( v24 != (__int64 *)(v23 + 56 * v22) )
              {
                v32 = 6;
                v33 = 0;
                v34 = v24[3];
                v26 = v34;
                if ( v34 != 0 && v34 != -4096 && v34 != -8192 )
                {
                  a2 = v24[1] & 0xFFFFFFFFFFFFFFF8LL;
                  sub_BD6050(&v32, a2);
                  v26 = v34;
                }
                if ( v26 && v26 != -8192 && v26 != -4096 )
                  goto LABEL_43;
                v32 = 6;
                v33 = 0;
                v34 = v24[6];
                v30 = v34;
                LOBYTE(a2) = v34 != -4096;
                if ( ((v34 != 0) & (unsigned __int8)a2) != 0 && v34 != -8192 )
                {
                  a2 = v24[4] & 0xFFFFFFFFFFFFFFF8LL;
                  sub_BD6050(&v32, a2);
                  v30 = v34;
                }
                if ( v30 && v30 != -8192 && v30 != -4096 )
                {
LABEL_43:
                  sub_BD60C0(&v32);
                  v27 = v24[6];
                  if ( v27 != 0 && v27 != -4096 && v27 != -8192 )
                    sub_BD60C0(v24 + 4);
                  v28 = v24[3];
                  LOBYTE(a2) = v28 != -4096;
                  if ( ((v28 != 0) & (unsigned __int8)a2) != 0 && v28 != -8192 )
                    sub_BD60C0(v24 + 1);
                  *v24 = -8192;
                  --*(_DWORD *)(a1 + 240);
                  ++*(_DWORD *)(a1 + 244);
                }
              }
            }
            else
            {
              v31 = 1;
              while ( v25 != -4096 )
              {
                a2 = ((_DWORD)v22 - 1) & (unsigned int)(v31 + a2);
                v24 = (__int64 *)(v23 + 56LL * (unsigned int)a2);
                v25 = *v24;
                if ( v14 == *v24 )
                  goto LABEL_36;
                ++v31;
              }
            }
          }
          v29 = v15 + 1;
          if ( v15 + 1 == v13 )
            break;
          v14 = *v29;
          for ( ++v15; *v29 >= 0xFFFFFFFFFFFFFFFELL; v15 = v29 )
          {
            if ( v13 == ++v29 )
              goto LABEL_21;
            v14 = *v29;
          }
        }
        while ( v13 != v15 );
      }
    }
LABEL_21:
    v16 = *(__int64 **)(a1 + 376);
    if ( *(_BYTE *)(a1 + 396) )
      v17 = *(unsigned int *)(a1 + 388);
    else
      v17 = *(unsigned int *)(a1 + 384);
    v18 = &v16[v17];
    if ( v16 != v18 )
    {
      v19 = *v16;
      v20 = *(__int64 **)(a1 + 376);
      if ( (unsigned __int64)*v16 < 0xFFFFFFFFFFFFFFFELL )
      {
LABEL_27:
        while ( v18 != v20 )
        {
          a2 = sub_ACADE0(*(__int64 ***)(v19 + 8));
          sub_BD84D0(v19, a2);
          sub_B43D60((_QWORD *)v19);
          v21 = v20 + 1;
          if ( v20 + 1 == v18 )
            break;
          while ( 1 )
          {
            v19 = *v21;
            v20 = v21;
            if ( (unsigned __int64)*v21 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v18 == ++v21 )
              goto LABEL_3;
          }
        }
      }
      else
      {
        while ( v18 != ++v16 )
        {
          v19 = *v16;
          v20 = v16;
          if ( (unsigned __int64)*v16 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_27;
        }
      }
    }
  }
LABEL_3:
  ++*(_QWORD *)(a1 + 256);
  if ( *(_BYTE *)(a1 + 284) )
    goto LABEL_8;
  v6 = 4 * (*(_DWORD *)(a1 + 276) - *(_DWORD *)(a1 + 280));
  v7 = *(unsigned int *)(a1 + 272);
  if ( v6 < 0x20 )
    v6 = 32;
  if ( v6 >= (unsigned int)v7 )
  {
    a2 = 0xFFFFFFFFLL;
    memset(*(void **)(a1 + 264), -1, 8 * v7);
LABEL_8:
    *(_QWORD *)(a1 + 276) = 0;
    goto LABEL_9;
  }
  sub_C8C990(a1 + 256, a2);
LABEL_9:
  ++*(_QWORD *)(a1 + 368);
  if ( !*(_BYTE *)(a1 + 396) )
  {
    v8 = 4 * (*(_DWORD *)(a1 + 388) - *(_DWORD *)(a1 + 392));
    v9 = *(unsigned int *)(a1 + 384);
    if ( v8 < 0x20 )
      v8 = 32;
    if ( v8 < (unsigned int)v9 )
    {
      sub_C8C990(a1 + 368, a2);
      return v5;
    }
    memset(*(void **)(a1 + 376), -1, 8 * v9);
  }
  *(_QWORD *)(a1 + 388) = 0;
  return v5;
}
