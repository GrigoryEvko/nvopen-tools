// Function: sub_22C4510
// Address: 0x22c4510
//
__int64 __fastcall sub_22C4510(__int64 a1, unsigned int a2)
{
  unsigned int v3; // ebx
  char v4; // dl
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  unsigned int v7; // r12d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r13
  bool v11; // zf
  _QWORD *v12; // r12
  unsigned __int64 *v13; // rbx
  __int64 v14; // r15
  unsigned __int64 *v15; // r15
  __int64 v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // r15
  __int64 v19; // rsi
  int v20; // ecx
  unsigned int v21; // eax
  _QWORD *v22; // r9
  __int64 v23; // rdx
  __int64 v24; // rbx
  int v25; // eax
  int v27; // r10d
  _QWORD *v28; // rdi
  __int64 v29; // rax
  _QWORD *v30; // r12
  _QWORD *v31; // r13
  unsigned __int64 *v32; // r15
  unsigned __int64 v33; // rax
  _QWORD *v34; // rdi
  int v35; // ecx
  __int64 v36; // rax
  __int64 v37; // [rsp+8h] [rbp-D8h]
  __int64 v38; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v39; // [rsp+18h] [rbp-C8h]
  __int64 v40; // [rsp+20h] [rbp-C0h]
  __int64 v41; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v42; // [rsp+38h] [rbp-A8h]
  __int64 v43; // [rsp+40h] [rbp-A0h]
  __int64 v44; // [rsp+50h] [rbp-90h] BYREF
  __int64 v45; // [rsp+58h] [rbp-88h]
  __int64 v46; // [rsp+60h] [rbp-80h]

  v3 = a2;
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4
    || (v5 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
                | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
                | (a2 - 1)
                | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
              | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
              | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
              | (a2 - 1)
              | ((unsigned __int64)(a2 - 1) >> 1)) >> 16)
            | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
              | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
              | (a2 - 1)
              | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
            | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
            | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
            | (a2 - 1)
            | ((unsigned __int64)(a2 - 1) >> 1))
           + 1,
        v3 = v5,
        (unsigned int)v5 > 0x40) )
  {
    if ( !v4 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      v37 = *(_QWORD *)(a1 + 16);
      if ( v3 <= 4 )
      {
        *(_BYTE *)(a1 + 8) |= 1u;
        goto LABEL_9;
      }
      v8 = 24LL * v3;
LABEL_5:
      v9 = sub_C7D670(v8, 8);
      *(_DWORD *)(a1 + 24) = v3;
      *(_QWORD *)(a1 + 16) = v9;
LABEL_9:
      v46 = -4096;
      v44 = 0;
      v45 = 0;
      v10 = 24LL * v7;
      v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
      *(_QWORD *)(a1 + 8) &= 1uLL;
      v12 = (_QWORD *)(v37 + v10);
      if ( v11 )
      {
        v13 = *(unsigned __int64 **)(a1 + 16);
        v14 = 3LL * *(unsigned int *)(a1 + 24);
      }
      else
      {
        v13 = (unsigned __int64 *)(a1 + 16);
        v14 = 12;
      }
      v15 = &v13[v14];
      if ( v15 != v13 )
      {
        do
        {
          if ( v13 )
          {
            *v13 = 0;
            v13[1] = 0;
            v16 = v46;
            v11 = v46 == 0;
            v13[2] = v46;
            if ( v16 != -4096 && !v11 && v16 != -8192 )
              sub_BD6050(v13, v44 & 0xFFFFFFFFFFFFFFF8LL);
          }
          v13 += 3;
        }
        while ( v15 != v13 );
        if ( v46 != 0 && v46 != -4096 && v46 != -8192 )
          sub_BD60C0(&v44);
      }
      v41 = 0;
      v17 = -4096;
      v42 = 0;
      v43 = -4096;
      v18 = (_QWORD *)v37;
      v44 = 0;
      v45 = 0;
      v46 = -8192;
      if ( v12 != (_QWORD *)v37 )
      {
        while ( 1 )
        {
          v24 = v18[2];
          if ( v24 != v17 )
          {
            v17 = v46;
            if ( v24 != v46 )
            {
              if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
              {
                v19 = a1 + 16;
                v20 = 3;
              }
              else
              {
                v25 = *(_DWORD *)(a1 + 24);
                v19 = *(_QWORD *)(a1 + 16);
                if ( !v25 )
                  BUG();
                v20 = v25 - 1;
              }
              v21 = v20 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
              v22 = (_QWORD *)(v19 + 24LL * v21);
              v23 = v22[2];
              if ( v23 != v24 )
              {
                v27 = 1;
                v28 = 0;
                while ( v23 != -4096 )
                {
                  if ( v23 != -8192 || v28 )
                    v22 = v28;
                  v21 = v20 & (v27 + v21);
                  v23 = *(_QWORD *)(v19 + 24LL * v21 + 16);
                  if ( v24 == v23 )
                    goto LABEL_24;
                  ++v27;
                  v28 = v22;
                  v22 = (_QWORD *)(v19 + 24LL * v21);
                }
                if ( v28 )
                {
                  v29 = v28[2];
                }
                else
                {
                  v29 = v22[2];
                  v28 = v22;
                }
                if ( v24 != v29 )
                {
                  if ( v29 != -4096 && v29 != 0 && v29 != -8192 )
                    sub_BD60C0(v28);
                  v28[2] = v24;
                  if ( v24 != -4096 && v24 != 0 && v24 != -8192 )
                    sub_BD73F0((__int64)v28);
                }
              }
LABEL_24:
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
              v17 = v18[2];
            }
          }
          if ( v17 != 0 && v17 != -4096 && v17 != -8192 )
            sub_BD60C0(v18);
          v18 += 3;
          if ( v12 == v18 )
            break;
          v17 = v43;
        }
        if ( v46 != 0 && v46 != -4096 && v46 != -8192 )
          sub_BD60C0(&v44);
        if ( v43 != -8192 && v43 != -4096 && v43 != 0 )
          sub_BD60C0(&v41);
      }
      return sub_C7D6A0(v37, v10, 8);
    }
    v38 = 0;
    v30 = (_QWORD *)(a1 + 16);
    v31 = (_QWORD *)(a1 + 112);
    v39 = 0;
    v40 = -4096;
    v41 = 0;
    v42 = 0;
    v43 = -8192;
  }
  else
  {
    if ( !v4 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(_DWORD *)(a1 + 24);
      v3 = 64;
      v8 = 1536;
      v37 = v6;
      goto LABEL_5;
    }
    v38 = 0;
    v30 = (_QWORD *)(a1 + 16);
    v31 = (_QWORD *)(a1 + 112);
    v3 = 64;
    v39 = 0;
    v40 = -4096;
    v41 = 0;
    v42 = 0;
    v43 = -8192;
  }
  v32 = (unsigned __int64 *)&v44;
  v33 = v30[2];
  if ( v33 == -4096 )
    goto LABEL_62;
LABEL_56:
  if ( v43 != v33 )
  {
    if ( v32 )
    {
      *v32 = 0;
      v32[1] = 0;
      v32[2] = v33;
      if ( v33 != 0 && v33 != -4096 && v33 != -8192 )
        sub_BD6050(v32, *v30 & 0xFFFFFFFFFFFFFFF8LL);
    }
    v32 += 3;
  }
LABEL_62:
  while ( 1 )
  {
    v34 = v30;
    v30 += 3;
    sub_D68D70(v34);
    if ( v30 == v31 )
      break;
    v33 = v30[2];
    if ( v33 != v40 )
      goto LABEL_56;
  }
  if ( v3 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v36 = sub_C7D670(24LL * v3, 8);
    *(_DWORD *)(a1 + 24) = v3;
    *(_QWORD *)(a1 + 16) = v36;
  }
  sub_22C41C0(a1, &v44, v32, v35);
  sub_D68D70(&v41);
  return sub_D68D70(&v38);
}
