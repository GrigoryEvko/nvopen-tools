// Function: sub_AFDBF0
// Address: 0xafdbf0
//
__int64 __fastcall sub_AFDBF0(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v3; // r13d
  __int64 v5; // r15
  _BYTE *v6; // r12
  _BYTE *v7; // r14
  __int64 v8; // rax
  unsigned int v9; // eax
  __int64 v10; // rdx
  unsigned int v11; // eax
  __int64 v12; // rdx
  unsigned int v13; // eax
  __int64 v14; // rdx
  unsigned int v15; // eax
  __int64 v16; // rdx
  unsigned int v17; // edx
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rdx
  int v21; // r15d
  unsigned int v22; // r15d
  __int64 *v23; // r13
  __int64 v24; // r15
  _BYTE *v25; // r14
  __int64 v26; // r12
  unsigned int v27; // eax
  __int64 v28; // rbx
  _BYTE *v29; // r9
  unsigned int v30; // eax
  __int64 v31; // rdx
  _BYTE *v32; // [rsp+8h] [rbp-108h]
  unsigned int v33; // [rsp+14h] [rbp-FCh]
  __int64 v34; // [rsp+18h] [rbp-F8h]
  __int64 *v35; // [rsp+20h] [rbp-F0h]
  __int64 v36; // [rsp+28h] [rbp-E8h]
  int v37; // [rsp+30h] [rbp-E0h]
  int v38; // [rsp+34h] [rbp-DCh]
  int v39; // [rsp+34h] [rbp-DCh]
  __int64 v41; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v42; // [rsp+48h] [rbp-C8h]
  _BYTE *v43; // [rsp+50h] [rbp-C0h]
  __int64 v44; // [rsp+58h] [rbp-B8h] BYREF
  __int64 v45; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v46; // [rsp+68h] [rbp-A8h] BYREF
  int v47; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v48; // [rsp+78h] [rbp-98h] BYREF
  int v49; // [rsp+80h] [rbp-90h]
  __int64 v50; // [rsp+88h] [rbp-88h]
  int v51; // [rsp+90h] [rbp-80h]
  int v52; // [rsp+94h] [rbp-7Ch]
  int v53; // [rsp+98h] [rbp-78h]
  int v54; // [rsp+9Ch] [rbp-74h]
  __int64 v55; // [rsp+A0h] [rbp-70h]
  __int64 v56; // [rsp+A8h] [rbp-68h]
  __int64 v57; // [rsp+B0h] [rbp-60h]
  __int64 v58; // [rsp+B8h] [rbp-58h]
  __int64 v59; // [rsp+C0h] [rbp-50h]
  __int64 v60; // [rsp+C8h] [rbp-48h]
  __int64 v61; // [rsp+D0h] [rbp-40h]

  v3 = *(_DWORD *)(a1 + 24);
  if ( v3 )
  {
    v5 = *a2;
    v6 = (_BYTE *)(*a2 - 16);
    v36 = *(_QWORD *)(a1 + 8);
    v7 = (_BYTE *)*((_QWORD *)sub_A17150(v6) + 1);
    v43 = v7;
    v44 = sub_AF5140(v5, 2u);
    v45 = sub_AF5140(v5, 3u);
    v8 = v5;
    if ( *(_BYTE *)v5 != 16 )
      v8 = *(_QWORD *)sub_A17150(v6);
    v46 = v8;
    v47 = *(_DWORD *)(v5 + 16);
    v48 = *((_QWORD *)sub_A17150(v6) + 4);
    v49 = *(_DWORD *)(v5 + 20);
    if ( (*(_BYTE *)(v5 - 16) & 2) != 0 )
      v9 = *(_DWORD *)(v5 - 24);
    else
      v9 = (*(_WORD *)(v5 - 16) >> 6) & 0xF;
    v10 = 0;
    if ( v9 > 8 )
      v10 = *((_QWORD *)sub_A17150(v6) + 8);
    v50 = v10;
    v51 = *(_DWORD *)(v5 + 24);
    v52 = *(_DWORD *)(v5 + 28);
    v53 = *(_DWORD *)(v5 + 32);
    v38 = *(_DWORD *)(v5 + 36);
    v54 = v38;
    v55 = *((_QWORD *)sub_A17150(v6) + 5);
    if ( (*(_BYTE *)(v5 - 16) & 2) != 0 )
      v11 = *(_DWORD *)(v5 - 24);
    else
      v11 = (*(_WORD *)(v5 - 16) >> 6) & 0xF;
    v12 = 0;
    if ( v11 > 9 )
      v12 = *((_QWORD *)sub_A17150(v6) + 9);
    v56 = v12;
    v57 = *((_QWORD *)sub_A17150(v6) + 6);
    v58 = *((_QWORD *)sub_A17150(v6) + 7);
    if ( (*(_BYTE *)(v5 - 16) & 2) != 0 )
      v13 = *(_DWORD *)(v5 - 24);
    else
      v13 = (*(_WORD *)(v5 - 16) >> 6) & 0xF;
    v14 = 0;
    if ( v13 > 0xA )
      v14 = *((_QWORD *)sub_A17150(v6) + 10);
    v59 = v14;
    if ( (*(_BYTE *)(v5 - 16) & 2) != 0 )
      v15 = *(_DWORD *)(v5 - 24);
    else
      v15 = (*(_WORD *)(v5 - 16) >> 6) & 0xF;
    v16 = 0;
    if ( v15 > 0xB )
      v16 = *((_QWORD *)sub_A17150(v6) + 11);
    v60 = v16;
    if ( (*(_BYTE *)(v5 - 16) & 2) != 0 )
      v17 = *(_DWORD *)(v5 - 24);
    else
      v17 = (*(_WORD *)(v5 - 16) >> 6) & 0xF;
    v18 = 0;
    if ( v17 > 0xC )
      v18 = sub_AF5140(v5, 0xCu);
    v61 = v18;
    v41 = 0;
    v42 = 0;
    if ( v7 )
    {
      if ( *v7 == 14 )
      {
        v19 = sub_AF5140((__int64)v7, 7u);
        if ( v19 )
        {
          v41 = sub_B91420(v19, 7);
          v42 = v20;
          LOBYTE(v38) = v54;
        }
      }
    }
    if ( (v38 & 8) == 0 && v45 && v43 && *v43 == 14 )
      v21 = sub_AFA7A0(&v45, &v41);
    else
      v21 = sub_AFA420(&v44, &v41, &v46, &v48, &v47);
    v22 = (v3 - 1) & v21;
    v39 = v3 - 1;
    v23 = (__int64 *)(v36 + 8LL * v22);
    if ( *a2 == *v23 )
    {
LABEL_57:
      *a3 = v23;
      return 1;
    }
    else
    {
      v37 = 1;
      v35 = 0;
      v33 = v22;
      v24 = *v23;
      v25 = (_BYTE *)(*a2 - 16);
      v26 = *a2;
      while ( v24 != -4096 )
      {
        if ( v24 == -8192 )
        {
          if ( !v35 )
            v35 = v23;
        }
        else
        {
          if ( (*(_BYTE *)(v26 - 16) & 2) != 0 )
            v27 = *(_DWORD *)(v26 - 24);
          else
            v27 = (*(_WORD *)(v26 - 16) >> 6) & 0xF;
          v34 = 0;
          if ( v27 > 9 )
            v34 = *((_QWORD *)sub_A17150(v25) + 9);
          v28 = sub_AF5140(v26, 3u);
          v29 = (_BYTE *)*((_QWORD *)sub_A17150(v25) + 1);
          if ( v29 != 0 && (*(_DWORD *)(v26 + 36) & 8) == 0 )
          {
            if ( v28 )
            {
              if ( *v29 == 14 )
              {
                v32 = v29;
                if ( sub_AF5140((__int64)v29, 7u) )
                {
                  if ( (*(_BYTE *)(v24 + 36) & 8) == 0
                    && v32 == *((_BYTE **)sub_A17150((_BYTE *)(v24 - 16)) + 1)
                    && v28 == sub_AF5140(v24, 3u) )
                  {
                    v30 = (*(_BYTE *)(v24 - 16) & 2) != 0 ? *(_DWORD *)(v24 - 24) : (*(_WORD *)(v24 - 16) >> 6) & 0xF;
                    v31 = 0;
                    if ( v30 > 9 )
                      v31 = *((_QWORD *)sub_A17150((_BYTE *)(v24 - 16)) + 9);
                    if ( v34 == v31 )
                      goto LABEL_57;
                  }
                }
              }
            }
          }
        }
        v33 = (v33 + v37) & v39;
        v23 = (__int64 *)(v36 + 8LL * v33);
        v24 = *v23;
        if ( *v23 == v26 )
          goto LABEL_57;
        ++v37;
      }
      if ( v35 )
        v23 = v35;
      *a3 = v23;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
