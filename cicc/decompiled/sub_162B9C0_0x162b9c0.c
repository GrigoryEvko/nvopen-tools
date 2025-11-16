// Function: sub_162B9C0
// Address: 0x162b9c0
//
__int64 __fastcall sub_162B9C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  int v3; // r12d
  char v4; // al
  _QWORD *v5; // rdx
  __int64 result; // rax
  int v7; // eax
  __int64 v8; // r9
  int v9; // r14d
  int v10; // eax
  int v11; // r12d
  __int64 v12; // r10
  unsigned int v13; // r11d
  unsigned int v14; // r14d
  __int64 *v15; // r15
  __int64 v16; // rsi
  __int64 v17; // r12
  __int64 v18; // r13
  unsigned int v19; // ebx
  char v20; // r14
  __int64 v21; // rdx
  __int64 v22; // r8
  __int64 v23; // r8
  char v24; // al
  unsigned int v25; // esi
  int v26; // eax
  int v27; // eax
  unsigned int v28; // [rsp+4h] [rbp-ECh]
  __int64 v29; // [rsp+8h] [rbp-E8h]
  __int64 v30; // [rsp+10h] [rbp-E0h]
  __int64 v31; // [rsp+20h] [rbp-D0h]
  int v32; // [rsp+28h] [rbp-C8h]
  int v33; // [rsp+2Ch] [rbp-C4h]
  __int64 v34; // [rsp+30h] [rbp-C0h]
  __int64 v35; // [rsp+30h] [rbp-C0h]
  __int64 v36; // [rsp+30h] [rbp-C0h]
  __int64 v37; // [rsp+38h] [rbp-B8h] BYREF
  __int64 v38; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v39; // [rsp+48h] [rbp-A8h] BYREF
  __int64 v40; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v41; // [rsp+58h] [rbp-98h] BYREF
  int v42; // [rsp+60h] [rbp-90h] BYREF
  __int64 v43; // [rsp+68h] [rbp-88h] BYREF
  char v44; // [rsp+71h] [rbp-7Fh]
  __int64 v45; // [rsp+A0h] [rbp-50h]

  v2 = a2;
  v37 = a1;
  sub_161E440((__int64)&v38, a1);
  v3 = *(_DWORD *)(a2 + 24);
  if ( !v3 )
    goto LABEL_2;
  if ( !v44 && v40 && v38 && *(_BYTE *)v38 == 13 && *(_QWORD *)(v38 + 8 * (7LL - *(unsigned int *)(v38 + 8))) )
  {
    v34 = *(_QWORD *)(a2 + 8);
    v7 = sub_15B2D00(&v40, &v38);
    v8 = v34;
    v9 = v7;
  }
  else
  {
    v35 = *(_QWORD *)(a2 + 8);
    v10 = sub_15B55D0(&v39, &v38, &v41, &v43, &v42);
    v8 = v35;
    v9 = v10;
  }
  v11 = v3 - 1;
  v12 = *(_QWORD *)(a2 + 8);
  v13 = *(_DWORD *)(a2 + 24);
  v14 = v11 & v9;
  v15 = (__int64 *)(v8 + 8LL * v14);
  v16 = *v15;
  if ( *v15 == -8 )
    goto LABEL_2;
  v33 = 1;
  v32 = v11;
  v31 = v45;
  v17 = v38;
  v36 = v2;
  v18 = v40;
  v19 = v14;
  v20 = v44 | (v40 == 0 || v38 == 0);
  while ( 1 )
  {
    if ( v16 != -16 )
    {
      if ( !v20 && *(_BYTE *)v17 == 13 )
      {
        if ( *(_QWORD *)(v17 + 8 * (7LL - *(unsigned int *)(v17 + 8))) )
        {
          if ( (*(_BYTE *)(v16 + 40) & 8) == 0 )
          {
            v21 = *(unsigned int *)(v16 + 8);
            if ( v17 == *(_QWORD *)(v16 + 8 * (1 - v21)) )
            {
              v22 = *(_QWORD *)(v16 + 8 * (3 - v21));
              if ( v22 )
              {
                if ( v18 == v22 )
                {
                  v23 = 0;
                  if ( (unsigned int)v21 > 9 )
                    v23 = *(_QWORD *)(v16 + 8 * (9 - v21));
                  if ( v31 == v23 )
                    break;
                }
              }
            }
          }
        }
      }
      v28 = v13;
      v29 = v8;
      v30 = v12;
      v24 = sub_15AFB30((__int64)&v38, v16);
      v12 = v30;
      v8 = v29;
      v13 = v28;
      if ( v24 )
        break;
    }
    v19 = v32 & (v33 + v19);
    v15 = (__int64 *)(v8 + 8LL * v19);
    v16 = *v15;
    if ( *v15 == -8 )
    {
      v2 = v36;
      goto LABEL_2;
    }
    ++v33;
  }
  v2 = v36;
  if ( v15 == (__int64 *)(v12 + 8LL * v13) || (result = *v15) == 0 )
  {
LABEL_2:
    v4 = sub_15B8340(v2, &v37, &v38);
    v5 = (_QWORD *)v38;
    if ( v4 )
      return v37;
    v25 = *(_DWORD *)(v2 + 24);
    v26 = *(_DWORD *)(v2 + 16);
    ++*(_QWORD *)v2;
    v27 = v26 + 1;
    if ( 4 * v27 >= 3 * v25 )
    {
      v25 *= 2;
    }
    else if ( v25 - *(_DWORD *)(v2 + 20) - v27 > v25 >> 3 )
    {
LABEL_30:
      *(_DWORD *)(v2 + 16) = v27;
      if ( *v5 != -8 )
        --*(_DWORD *)(v2 + 20);
      *v5 = v37;
      return v37;
    }
    sub_15BFA40(v2, v25);
    sub_15B8340(v2, &v37, &v38);
    v5 = (_QWORD *)v38;
    v27 = *(_DWORD *)(v2 + 16) + 1;
    goto LABEL_30;
  }
  return result;
}
